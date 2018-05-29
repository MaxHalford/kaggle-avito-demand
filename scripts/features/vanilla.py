import re
import string

import numpy as np
import pandas as pd
import xam


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

# Last digit in price
data['price_last_digit'] = data['price'].map(
    lambda x: str(int(x))[-1]
    if np.isfinite(x)
    else -1
)

# Text features
data['description'] = data['description'].fillna('')
punctuation = set(string.punctuation)


def is_emoji(c):
    return not(c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punctuation)


emojis = set()
for s in pd.concat((data['title'], data['description'])):
    for c in s:
        if is_emoji(c):
            emojis.add(c)

data['n_title_chars'] = data['title'].apply(len)
data['n_title_words'] = data['title'].apply(lambda x: len(re.findall(r'\w+', x)))
data['n_title_digits'] = data['title'].apply(lambda x: sum(c.isdigit() for c in x))
data['n_title_upper'] = data['title'].apply(lambda x: sum(c.isupper() for c in x))
data['n_title_spaces'] = data['title'].apply(lambda x: sum(c.isspace() for c in x))
data['n_title_punct'] = data['title'].apply(lambda x: sum(c in punctuation for c in x))
data['n_title_emoji'] = data['title'].apply(lambda x: sum(c in emojis for c in x))

data['r_title_words'] = data['n_title_words'] / (data['n_title_chars'] + 1)
data['r_title_digits'] = data['n_title_digits'] / (data['n_title_chars'] + 1)
data['r_title_upper'] = data['n_title_upper'] / (data['n_title_chars'] + 1)
data['r_title_spaces'] = data['n_title_spaces'] / (data['n_title_chars'] + 1)
data['r_title_punct'] = data['n_title_punct'] / (data['n_title_chars'] + 1)
data['r_title_emoji'] = data['n_title_emoji'] / (data['n_title_chars'] + 1)

data['n_description_chars'] = data['description'].apply(len)
data['n_description_words'] = data['description'].apply(lambda x: len(re.findall(r'\w+', x)))
data['n_description_digits'] = data['description'].apply(lambda x: sum(c.isdigit() for c in x))
data['n_description_upper'] = data['description'].apply(lambda x: sum(c.isupper() for c in x))
data['n_description_spaces'] = data['description'].apply(lambda x: sum(c.isspace() for c in x))
data['n_description_punct'] = data['description'].apply(lambda x: sum(c in punctuation for c in x))
data['n_description_emoji'] = data['description'].apply(lambda x: sum(c in emojis for c in x))

data['r_description_words'] = data['n_description_words'] / (data['n_description_chars'] + 1)
data['r_description_digits'] = data['n_description_digits'] / (data['n_description_chars'] + 1)
data['r_description_upper'] = data['n_description_upper'] / (data['n_description_chars'] + 1)
data['r_description_spaces'] = data['n_description_spaces'] / (data['n_description_chars'] + 1)
data['r_description_punct'] = data['n_description_punct'] / (data['n_description_chars'] + 1)
data['r_description_emoji'] = data['n_description_emoji'] / (data['n_description_chars'] + 1)

# Difference with mean price per category and param
data['id'] = data['category_name'] + data['param_1'] + data['param_2'] + data['param_3']
mean_prices = data.groupby('id')['price'].mean().map(np.log1p).to_dict()
data['price_diff'] = (data['price'].map(np.log1p) - data['id'].map(mean_prices)).abs()

# Clean up missing values
data['price'] = data['price'].fillna(-1)
data['param_1'] = data['param_1'].fillna('missing')

# Target encoding
encoder = xam.feature_extraction.SmoothTargetEncoder(
    columns=['city', 'region', 'category_name', 'parent_category_name', 'param_1', 'id'],
    prior_weight=500,
    suffix=''
)
data = encoder.fit_transform(data, data['deal_probability'])

# Drop unneeded columns
cols_to_drop = ['title', 'description', 'activation_date', 'param_2', 'param_3']
data.drop(cols_to_drop, axis='columns', inplace=True)

# Save the features
train_mask = data['deal_probability'].notnull()
data[train_mask].to_csv('features/train/vanilla.csv', index=False)
data[~train_mask].to_csv('features/test/vanilla.csv', index=False)
