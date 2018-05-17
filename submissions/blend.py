import pandas as pd
import glob


files = glob.glob('*.csv')

sub = pd.read_csv(files[0])
sub['deal_probability'] = 1

weights = {
    'stacking_0.21495_0.00073.csv': 2,
    'submission.csv': 1,
    'blend 06.csv': 1
}

for f in files:
    print(f)
    weight = weights[f]
    sub['deal_probability'] *= pd.read_csv(f)['deal_probability'] ** weight

sub['deal_probability'] = sub['deal_probability'] ** (1 / sum(weights.values()))

sub.to_csv('blend.csv', index=False)
