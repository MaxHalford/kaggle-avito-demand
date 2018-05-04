import pandas as pd
import glob


files = glob.glob('*.csv')

sub = pd.read_csv(files[0])
sub['deal_probability'] = 1

for f in files:
    print(f)
    sub['deal_probability'] *= pd.read_csv(f)['deal_probability']

sub['deal_probability'] = sub['deal_probability'] ** (1 / len(files))

sub.to_csv('blend.csv', index=False)
