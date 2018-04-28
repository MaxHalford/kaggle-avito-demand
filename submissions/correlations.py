import glob

import pandas as pd


correlations = []

files = glob.glob('*.csv')

for i, file1 in enumerate(files):
    for file2 in files[i+1:]:
        correlations.append((
            file1,
            file2,
            pd.read_csv(file1)['deal_probability'].corr(pd.read_csv(file2)['deal_probability'])
        ))

print(correlations)
