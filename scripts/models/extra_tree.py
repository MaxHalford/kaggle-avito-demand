import json
import os

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor


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


def preprocessing_sklearn(train, test):
    """Preprocessing for sklearn input.
    Parameters
    ----------
    train : type
        DataFrame.
    test : type
        DataFrame.

    Returns
    -------
    type
        (DataFrame,DataFrame).

    """

    data = pd.concat([train, test], axis='rows')
    train_mask = data['deal_probability'].notnull()

    data['price'] = data.groupby('category_name')[
        'price'].transform(lambda x: x.fillna(x.mean()))

    #data['category_price_diff'].fillna(0, inplace=True)
    img_fill = ['image_top_1', 'n_pixels',
                'sharpness', 'brightness', 'contrast']
    for img in img_fill:
        data[img].fillna(0, inplace=True)

    #boolean_col = ['no_price', 'title_in_sup', 'round_price', 'user_id_in_sup']
    boolean_col = ['title_in_sup', 'round_price', 'user_id_in_sup']
    data[boolean_col] = data[boolean_col].astype(int)

    data = pd.get_dummies(
        data, columns=['category_name', 'parent_category_name', 'user_type'])
    param_to_fill = ['param_1', 'param_2', 'param_3']

    for param in param_to_fill:

        data[param].fillna('Nujabes we miss you', inplace=True)
        data = data.join(data.groupby(param)['deal_probability'].mean().rename(
            param + '_mean'), on=param)
        data.drop([param], axis='columns', inplace=True)
        # we still have 76 nan so we replace them with the mean
        data[param + '_mean'].fillna(data[param +
                                          '_mean'].mean(), inplace=True)
    user_features_to_fill = ['avg_days_up_user',
                             'avg_times_up_user', 'n_user_items']
    for feature in user_features_to_fill:
        data[feature].fillna(data[feature].median(), inplace=True)
    return data[train_mask], data[~train_mask]

# Load train/test features
train = load_data('features/train')
test = load_data('features/test')
train, test = preprocessing_sklearn(train, test)

X_train = train.drop(['deal_probability', 'image'], axis='columns')
y_train = train['deal_probability']

X_test = test.drop(['deal_probability', 'image', 'item_id'], axis='columns')


sub = test[['item_id', 'deal_probability']].copy()
sub['deal_probability'] = 1


# Load folds
with open('folds/folds_item_ids.json') as infile:
    folds_item_ids = json.load(infile)

fit_scores = []
val_scores = []


def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


model = ExtraTreesRegressor(n_estimators=100, max_features=0.7, max_depth=10)


for i in folds_item_ids.keys():

    # Determine train and val folds
    fit_mask = X_train['item_id'].isin(folds_item_ids[i]['fit'])
    val_mask = X_train['item_id'].isin(folds_item_ids[i]['val'])
    X_fit = X_train[fit_mask].drop('item_id', axis='columns')
    y_fit = y_train[fit_mask]
    X_val = X_train[val_mask].drop('item_id', axis='columns')
    y_val = y_train[val_mask]

    # trick for ram saving
    model.fit(X_fit.astype(dtype='float32'), y_fit.astype(dtype='float32'))

    fit_predict = model.predict(X_fit)
    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)
    fit_scores.append(rmse(y_fit, fit_predict))
    val_scores.append(rmse(y_val, val_predict))
    sub['deal_probability'] *= test_predict

    # Save out-of-fold predictions
    name = 'folds/extra_tree_val_{}.csv'.format(i)
    pd.Series(val_predict).to_csv(name, index=False)
    # Save test predictions
    name = 'folds/extra_tree_test_{}.csv'.format(i)
    pd.Series(test_predict).to_csv(name, index=False)

    print('Fold {} val RMSE: {:.5f}'.format(int(i) + 1, val_scores[int(i)]))
    print('Fold {} fit RMSE: {:.5f}'.format(int(i) + 1, fit_scores[int(i)]))

# Show train and validation scores
fit_mean = np.mean(fit_scores)
fit_std = np.std(fit_scores)
val_mean = np.mean(val_scores)
val_std = np.std(val_scores)
print('Fit RMSE: {:.5f} (±{:.5f})'.format(fit_mean, fit_std))
print('Val RMSE: {:.5f} (±{:.5f})'.format(val_mean, val_std))


# Save submission

sub['deal_probability'] = (sub['deal_probability'] **
                           (1 / len(folds_item_ids))).clip(0, 1)
sub_name = 'submissions/extra_tree_test_{:.5f}_{:.5f}_{:.5f}_{:.5f}.csv'.format(
    fit_mean,
    fit_std,
    val_mean,
    val_std


)

sub.to_csv(sub_name, index=False)
