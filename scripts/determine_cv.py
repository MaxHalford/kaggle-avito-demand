import json

import pandas as pd
from sklearn import model_selection


N_SPLITS = 5
SHUFFLE = True
RANDOM_STATE = 42


if __name__ == '__main__':

    train = pd.read_csv('data/train.csv.zip', usecols=['item_id'])

    cv = model_selection.KFold(
        n_splits=N_SPLITS,
        shuffle=SHUFFLE,
        random_state=RANDOM_STATE
    )

    folds = {}

    for i, (fit_idx, val_idx) in enumerate(cv.split(train)):
        fit = train.iloc[fit_idx]
        val = train.iloc[val_idx]
        folds[i] = {}
        folds[i]['fit'] = fit['item_id'].values.tolist()
        folds[i]['val'] = val['item_id'].values.tolist()

    with open('folds/folds_item_ids.json', 'w') as outfile:
        json.dump(folds, outfile)
