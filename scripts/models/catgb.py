import json
import os

import catboost
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing


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
            'param_1': 'category',
            'param_2': 'category',
            'param_3': 'category'
        }
    )

    # Add image quality
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(path_prefix, 'image_quality.csv')),
        how='left',
        on='image'
    )

    # Add title SVD components
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(path_prefix, 'title_svd.csv')),
        how='left',
        on='item_id'
    )

    # Add description SVD components
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(path_prefix, 'description_svd.csv')),
        how='left',
        on='item_id'
    )

    # Add active features
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(path_prefix, 'active.csv')),
        how='left',
        on='item_id'
    )

    # Add title probas
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(path_prefix, 'title_likelihoods.csv')),
        how='left',
        on='item_id'
    )

    # Add description probas
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(
            path_prefix, 'description_likelihoods.csv')),
        how='left',
        on='item_id'
    )

    # Add city geocode
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(path_prefix, 'geocode.csv')),
        how='left',
        on='item_id'
    )

    data = pd.merge(
        left=data,
        right=pd.read_csv('features/aggregated_features.csv'),
        how='left',
        on='user_id')
    return data


# Load train features
train = load_data('features/train')
X_train = train.drop(['deal_probability', 'image'], axis='columns')
y_train = train['deal_probability']

# Load test features
test = load_data('features/test')
sub = test[['item_id', 'deal_probability']].copy()
sub['deal_probability'] = 0
X_test = test.drop(['deal_probability', 'image', 'item_id'], axis='columns')

# Determine the categorical features
columns = X_test.columns
cat_columns = X_test.select_dtypes(include=['object']).columns
cat_features = []
for i, col in enumerate(X_test.columns):
    if col in cat_columns:
        cat_features.append(i)
        X_train[col] = X_train[col].fillna('nujabes')
        X_test[col] = X_test[col].fillna('nujabes')
X_test = catboost.Pool(X_test, cat_features=cat_features)

# Load folds
with open('folds/folds_item_ids.json') as infile:
    folds_item_ids = json.load(infile)

fit_scores = {}
val_scores = {}


def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


model = catboost.CatBoostRegressor(
    iterations=2000,
    learning_rate=0.5,
    max_depth=6,
    use_best_model=True,
    loss_function='RMSE',
    eval_metric='RMSE',
    od_type='Iter',
    od_wait=20,
    logging_level='Verbose',
    random_seed=42,
    boosting_type='Plain',
    one_hot_max_size=50
)


for i in folds_item_ids.keys():

    # Determine train and val folds
    fit_mask = X_train['item_id'].isin(folds_item_ids[i]['fit'])
    val_mask = X_train['item_id'].isin(folds_item_ids[i]['val'])
    X_fit = X_train[fit_mask].drop('item_id', axis='columns')
    y_fit = y_train[fit_mask]
    X_val = X_train[val_mask].drop('item_id', axis='columns')
    y_val = y_train[val_mask]
    fit = catboost.Pool(X_fit, y_fit, cat_features=cat_features)
    val = catboost.Pool(X_val, y_val, cat_features=cat_features)

    model.fit(
        fit,
        eval_set=val
    )

    fit_predict = model.predict(fit)
    val_predict = model.predict(val)
    test_predict = model.predict(X_test)
    fit_scores[i] = rmse(y_fit, fit_predict)
    val_scores[i] = rmse(y_val, val_predict)
    sub['deal_probability'] += test_predict

    # Save out-of-fold predictions
    name = 'folds/catboost_val_{}.csv'.format(i)
    pd.Series(val_predict).to_csv(name, index=False)
    # Save test predictions
    name = 'folds/catboost_test_{}.csv'.format(i)
    pd.Series(test_predict).to_csv(name, index=False)

    print('Fold {} RMSE: {:.5f}'.format(int(i) + 1, val_scores[i]))

# Show train and validation scores
fit_mean = np.mean(list(fit_scores.values()))
fit_std = np.std(list(fit_scores.values()))
val_mean = np.mean(list(val_scores.values()))
val_std = np.std(list(val_scores.values()))
print('Fit RMSE: {:.5f} (±{:.5f})'.format(fit_mean, fit_std))
print('Val RMSE: {:.5f} (±{:.5f})'.format(val_mean, val_std))

# Save submission
sub['deal_probability'] = (sub['deal_probability'] /
                           len(folds_item_ids)).clip(0, 1)
sub_name = 'submissions/lgbm_{:.5f}_{:.5f}_{:.5f}_{:.5f}.csv'.format(
    fit_mean,
    fit_std,
    val_mean,
    val_std
)
sub.to_csv(sub_name, index=False)
