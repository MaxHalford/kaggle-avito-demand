import json
import os

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import ElasticNet


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

    # Add number of pixels per image
    data = pd.merge(
        left=data,
        right=pd.read_csv(os.path.join(path_prefix, 'n_pixels.csv')),
        how='left',
        on='image'
    )
    data['n_pixels'].fillna(-1, inplace=True)

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
    return data


def preprocessing_sklearn(df):
    

    return df
# Load train features
train = load_data('features/train')
train = preprocessing_sklearn(train)
X_train = train.drop(['deal_probability', 'image'], axis='columns')
y_train = train['deal_probability']

# Load test features
test = load_data('features/test')
test = preprocessing_sklearn(test)
sub = test[['item_id', 'deal_probability']].copy()
sub['deal_probability'] = 0
X_test = test.drop(['deal_probability', 'image', 'item_id'], axis='columns')

# Load folds
with open('folds/folds_item_ids.json') as infile:
    folds_item_ids = json.load(infile)

fit_scores = []
val_scores = []


def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


model = ElasticNet(alpha=1.0, l1_ratio=0)


for i in folds_item_ids.keys():

    # Determine train and val folds
    fit_mask = X_train['item_id'].isin(folds_item_ids[i]['fit'])
    val_mask = X_train['item_id'].isin(folds_item_ids[i]['val'])
    X_fit = X_train[fit_mask].drop('item_id', axis='columns')
    y_fit = y_train[fit_mask]
    X_val = X_train[val_mask].drop('item_id', axis='columns')
    y_val = y_train[val_mask]

    model.fit(X_fit, y_fit)

    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)
    fit_scores.append(rmse(y_fit, val_predict))
    val_scores.append(rmse(y_val, test_predict))
    sub['deal_probability'] += test_predict

    # Save out-of-fold predictions
    name = 'folds/elasticnet_lr_val_{}.csv'.format(i)
    pd.Series(val_predict).to_csv(name, index=False)
    # Save test predictions
    name = 'folds/elasticnet_lr_test_{}.csv'.format(i)
    pd.Series(test_predict).to_csv(name, index=False)

    print('Fold {} RMSE: {:.5f}'.format(int(i) + 1, val_scores[i]))

# Show train and validation scores
fit_mean = np.mean(fit_scores.values())
fit_std = np.std(fit_scores.values())
val_mean = np.mean(val_scores.values())
val_std = np.std(val_scores.values())
print('Fit RMSE: {:.5f} (±{:.5f})'.format(fit_mean, fit_std))
print('Val RMSE: {:.5f} (±{:.5f})'.format(val_mean, val_std))


# Save submission
sub['deal_probability'] = (sub['deal_probability'] /
                           len(folds_item_ids)).clip(0, 1)
sub_name = 'submissions/elasticnet_lr_test_{:.5f}_{:.5f}_{:.5f}_{:.5f}.csv'.format(
    fit_mean,
    fit_std,
    val_mean,
    val_std
)

sub.to_csv(sub_name, index=False)
