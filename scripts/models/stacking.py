import glob
import pandas as pd


model_prefixes = [
    'lgbm',
    'cnn_title'
]

val = pd.DataFrame()
test = pd.DataFrame()

for model in model_prefixes:

    val_preds = []
    test_preds = []

    for f in sorted(glob.glob('folds/{}_val_*.csv'.format(model))):
        print(f)
        val_preds.extend(pd.read_csv(f, header=None)[0])

    for f in sorted(glob.glob('folds/{}_test_*.csv'.format(model))):
        test_preds.extend(pd.read_csv(f, header=None)[0])

    val[model] = val_preds
    test[model] = test_preds

print(val)
