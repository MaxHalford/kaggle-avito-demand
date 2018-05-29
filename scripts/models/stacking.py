import glob
import json

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import isotonic
from sklearn import metrics
from sklearn import model_selection


def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


# Choose which models to use
model_names = [
    'lgbm_2',
    'lgbm_6',
    'lgbm_8',
    'lgbm_9',
    'lgbm_10',
    'lgbm_13',
    'catboost',
    'cnn_title',
    'extra_tree',
    'max_means',
    'max_lgbm',
    'max_nn',
    'NN',
    'xgb_2',
    'xgb_6',
    'xgb_10'
]

# Load the true labels in the right order
item_ids = []
with open('folds/folds_item_ids.json') as infile:
    for fold in json.load(infile).values():
        item_ids.extend(fold['val'])

y_train = pd.read_csv('data/train.csv.zip')[['item_id', 'deal_probability']]\
            .set_index('item_id')\
            .loc[item_ids]['deal_probability']

# Load the out-of-fold predictions
X_train = pd.DataFrame()
X_test = pd.DataFrame()

# Applying isotonic regression to each set of predictions helps a wee bit
iso = isotonic.IsotonicRegression(y_min=0, y_max=1)

for model_name in model_names:

    y_pred_train = []
    y_pred_test = []

    for f in sorted(glob.glob('folds/{}_val_*.csv'.format(model_name))):
        y_pred_train.extend(pd.read_csv(f, header=None)[0].clip(0, 1))

    for f in sorted(glob.glob('folds/{}_test_*.csv'.format(model_name))):
        y_pred_test.append(pd.read_csv(f, header=None)[0].clip(0, 1))

    y_pred_train = np.array(y_pred_train)
    iso.fit(y_pred_train, y_train)

    X_train[model_name] = iso.transform(y_pred_train)
    X_test[model_name] = iso.transform(np.array(y_pred_test).mean(axis=0))

    print('{} has an average RMSE of {:.5f}'.format(model_name, rmse(y_train, X_train[model_name])))


# Choose meta-model
depth = 4
meta_model = lgbm.LGBMRegressor(
    objective='regression',
    n_estimators=30000,
    learning_rate=0.05,
    num_leaves=2 ** depth,
    subsample=0.8,
    colsample_bytree=0.8,
    bagging_seed=42,
    verbose=1
)

# Determine the CV score of the meta-model

n_splits = 5
val_scores = [0] * n_splits

sub = pd.read_csv('data/test.csv.zip')[['item_id']]
sub['deal_probability'] = 0

feature_importances = pd.DataFrame(index=X_train.columns)

cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)

for i, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):

    X_fit = X_train.iloc[fit_idx]
    y_fit = y_train.iloc[fit_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    evals_result = {}
    meta_model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_val, y_val)],
        eval_names=('fit', 'val'),
        eval_metric='l2',
        feature_name=X_fit.columns.tolist(),
        early_stopping_rounds=50
    )

    val_scores[i] = np.sqrt(meta_model.best_score_['val']['l2'])
    sub['deal_probability'] += meta_model.predict(X_test, num_iteration=meta_model.best_iteration_)
    feature_importances[i] = meta_model.feature_importances_


sub['deal_probability'] = (sub['deal_probability'] / n_splits).clip(0, 1)

val_mean = np.mean(val_scores)
val_std = np.std(val_scores)

print('Local RMSE: {:.5f} (±{:.5f})'.format(val_mean, val_std))

sub.to_csv('submissions/stacking_{:.5f}_{:.5f}.csv'.format(val_mean, val_std), index=False)
