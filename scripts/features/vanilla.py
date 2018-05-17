import re
import string

import numpy as np
import pandas as pd


data = pd.concat(
    (
        pd.read_csv('data/train.csv.zip', parse_dates=['activation_date']),
        pd.read_csv('data/test.csv.zip', parse_dates=['activation_date'])
    ),
    ignore_index=True
)

# Number of null values per row
data['n_missing'] = data.isnull().sum(axis='columns').astype('uint8')

# Activation date day of the week
data['activation_dow'] = data['activation_date'].dt.dayofweek

# Is the price a round price (e.g. 100 or 1000, not 42 or 1337)
data['round_price'] = data['price'].map(
    lambda x: str(int(x)).endswith('0')
    if np.isfinite(x)
    else False
)

# Number of characters in the description
data['desc_n_characters'] = data['description'].fillna('').map(len).astype('uint')

# Number of punctuation symbols in the description
data['desc_n_punctuation'] = data['description'].fillna('').apply(lambda x: sum(1 for c in x if c in string.punctuation))

# Percentage of punctuation symbols in the description
data['desc_punctuation_ratio'] = (data['desc_n_punctuation'] / data['desc_n_characters']).fillna(0)

# Percentage of uppercase letters in the description
data['desc_upper_ratio'] = data['description'].fillna('')\
                                                     .str.replace(' ', '')\
                                                     .map(lambda x: sum(l.isupper() for l in x) / (len(x) + 1))

# Number of words in the description
data['desc_n_words'] = data['description'].fillna('').map(lambda x: len(re.findall(r'\w+', x))).astype('uint')

# Number of characters in the title
data['title_n_characters'] = data['title'].map(len).astype('uint')

# Number of punctuation symbols in the title
data['title_n_punctuation'] = data['title'].apply(lambda x: sum(1 for c in x if c in string.punctuation))

# Percentage of punctuation symbols in the title
data['title_punctuation_ratio'] = data['title_n_punctuation'] / data['title_n_characters']

# Percentage of uppercase letters in the title
data['title_upper_ratio'] = data['title'].str.replace(' ', '')\
                                         .map(lambda x: sum(l.isupper() for l in x) / (len(x) + 1))

# Number of words in the title
data['title_n_words'] = data['title'].map(lambda x: len(re.findall(r'\w+', x))).astype('uint')

# Drop unneeded columns
cols_to_drop = ['title', 'description', 'activation_date',
                'region', 'city']
data.drop(cols_to_drop, axis='columns', inplace=True)

# Save the features
train_mask = data['deal_probability'].notnull()
data[train_mask].to_csv('features/train/vanilla.csv', index=False)
data[~train_mask].to_csv('features/test/vanilla.csv', index=False)
