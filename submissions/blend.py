import pandas as pd
import glob


files = glob.glob('*.csv')

sub = pd.read_csv(files[0])
sub['deal_probability'] = 1

weights = {
    'stacking_0.21375_0.00018.csv': 1.3,
    'submission.csv': 1.4,
    'blend.csv': 1.3,
    'blend 06.csv': 1.2,
    'lgsub.csv': 0.8
}

for f in files:
    print(f)
    weight = weights[f]
    sub['deal_probability'] *= (pd.read_csv(f)['deal_probability'].clip(0, 1) ** weight)

sub['deal_probability'] = sub['deal_probability'] ** (1 / sum(weights.values()))

sub.to_csv('blend.csv', index=False)
