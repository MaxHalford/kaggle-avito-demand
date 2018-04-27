import os

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection


# Yo, science, bitch
np.random.seed(42 * 1337)


def load_data(path_prefix):

    # Load vanilla features
    data = pd.read_csv(
        os.path.join(path_prefix, 'vanilla.csv'),
        dtype={
            'category_name': 'category',
            'parent_category_name': 'category',
            'user_type': 'category',
            'user_id': 'category'
        }
    )

    # Add number of pixels per image
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(path_prefix, 'n_pixels.csv')),
        how='left',
        on='image'
    )
    data['n_pixels'].fillna(-1, inplace=True)

    return data


# Load train features
train = load_data('features/train')
X_train = train.drop(['deal_probability', 'item_id', 'image'], axis='columns')
y_train = train['deal_probability']

# Load test features
test = load_data('features/test')
sub = test[['item_id', 'deal_probability']].copy()
sub['deal_probability'] = 0
X_test = test.drop(['deal_probability', 'item_id', 'image'], axis='columns')

n_splits = 5
cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)

fit_scores = [0] * n_splits
val_scores = [0] * n_splits
feature_importances = pd.DataFrame(index=X_train.columns)
sub['deal_probability'] = 0


def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


def rmse_lgbm(y_pred, dataset):
    """Custom RMSE evaluation metric tailored for LightGBM."""
    return ('RMSE', rmse(dataset.get_label(), y_pred), False)


params = {
    'application': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'None',
    'max_depth': 5,
    'num_leaves': 5 ** 2 - 1,
    'min_data_in_leaf': 10,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 1,
    'verbosity': 1
}


for i, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):

    fit = lgbm.Dataset(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    val = lgbm.Dataset(X_train.iloc[val_idx], y_train.iloc[val_idx])

    model = lgbm.train(
        params,
        fit,
        num_boost_round=30000,
        valid_sets=(fit, val),
        valid_names=('fit', 'val'),
        feval=rmse_lgbm,
        verbose_eval=50,
        early_stopping_rounds=20
    )

    fit_scores[i] = rmse(fit.get_label(), model.predict(fit))
    val_scores[i] = rmse(val.get_label(), model.predict(val))
    feature_importances[i] = model.feature_importance
    sub['deal_probability'] += model.predict(X_test)

    print('Fold {} RMSE: {:.5f}'.format(i+1, val_scores[i]))

# Show train and validation scores
fit_mean = np.mean(fit_scores)
fit_std = np.std(fit_scores)
val_mean = np.mean(val_scores)
val_std = np.std(val_scores)
print('Fit RMSE: {:.5f} (±{:.5f})'.format(fit_mean, fit_std))
print('Val RMSE: {:.5f} (±{:.5f})'.format(val_mean, val_std))

# Save feature_importances
feature_importances.to_csv('feature_importances.csv')

# Save submission
sub['deal_probability'] = (sub['deal_probability'] / n_splits).clip(0, 1)
sub_name = 'submissions/lgbm_vanilla_{:.5f}_{:.5f}_{:.5f}_{:.5f}.csv'.format(
    fit_mean,
    fit_std,
    val_mean,
    val_std
)
sub.to_csv(sub_name, index=False)
