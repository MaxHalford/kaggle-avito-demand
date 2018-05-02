import glob
import json

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection


def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


# Choose which models to use
model_names = [
    'lgbm',
    'random_forest'
]

# Load the true labels in the right order
item_ids = []
with open('folds/folds_item_ids.json') as infile:
    for fold in json.load(infile).values():
        item_ids.extend(fold['val'])

y_train = pd.read_csv('data/train.csv.zip')[['item_id', 'deal_probability']]\
            .set_index('item_id')\
            .loc[item_ids]

# Load the out-of-fold predictions
X_train = pd.DataFrame()
X_test = pd.DataFrame()

for model_name in model_names:

    y_pred_train = []
    y_pred_test = []

    for f in sorted(glob.glob('folds/{}_val_*.csv'.format(model_name))):
        y_pred_train.extend(pd.read_csv(f, header=None)[0])

    for f in sorted(glob.glob('folds/{}_test_*.csv'.format(model_name))):
        y_pred_test.append(pd.read_csv(f, header=None)[0])

    X_train[model_name] = y_pred_train
    X_test[model_name] = np.array(y_pred_test).mean(axis=0)

    print('{} has an average RMSE of {:.5f}'.format(model_name, rmse(y_pred_train, y_train)))

# Choose meta-model
meta_model = linear_model.LinearRegression()

# Determine the CV score of the meta-model for the lolz
cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
scores = model_selection.cross_val_score(
    meta_model,
    X_train,
    y_train,
    scoring=metrics.make_scorer(rmse)
)
print('Meta-model RMSE: {:.5f} (±{:.5f})'.format(scores.mean(), scores.std()))

# Train the model on all the data
meta_model.fit(X_train, y_train)
print('Intercept: {:.5f}'.format(meta_model.intercept_[0]))
for i, model_name in enumerate(model_names):
    print('{} coefficient: {:.5f}'.format(model_name, meta_model.coef_[0][i]))

# Make final predictions
test = pd.read_csv('data/test.csv.zip')[['item_id']]
test['deal_probability'] = meta_model.predict(X_test).clip(0, 1)
test.to_csv('submissions/stacking_{:.5f}_{:.5f}.csv'.format(scores.mean(), scores.std()), index=False)
