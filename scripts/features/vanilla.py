import re

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

# Number of adds per user
data = data.join(data.groupby('user_id').size().rename('user_n_ads').astype('uint'), on='user_id')

# Number of adds per city
data = data.join(data.groupby('city').size().rename('city_n_ads').astype('uint'), on='city')

# Number of adds per region
data = data.join(data.groupby('region').size().rename('region_n_ads').astype('uint'), on='region')

# Number of adds per title
data = data.join(data.groupby('title').size().rename('title_n_ads').astype('uint'), on='title')

# Activation date day of the week
data['activation_dow'] = data['activation_date'].dt.dayofweek

# No price is indicated
data['no_price'] = data['price'].isnull()

# Is the price a round price (e.g. 100 or 1000, not 42 or 1337)
data['round_price'] = data['price'].map(
    lambda x: str(int(x)).endswith('0')
    if np.isfinite(x)
    else False
)

# Number of characters in the description
data['description_n_characters'] = data['description'].fillna('').map(len).astype('uint')

# Percentage of uppercase letters in the description
data['description_upper_ratio'] = data['description'].fillna('')\
                                                     .str.replace(' ', '')\
                                                     .map(lambda x: sum(l.isupper() for l in x) / (len(x) + 1))

# Number of words in the description
data['description_n_words'] = data['description'].fillna('').map(lambda x: len(re.findall(r'\w+', x))).astype('uint')

# Number of characters in the title
data['title_n_characters'] = data['title'].fillna('').map(len).astype('uint')

# Percentage of uppercase letters in the title
data['title_upper_ratio'] = data['title'].fillna('')\
                                         .str.replace(' ', '')\
                                         .map(lambda x: sum(l.isupper() for l in x) / (len(x) + 1))

# Number of words in the title
data['title_n_words'] = data['title'].fillna('').map(lambda x: len(re.findall(r'\w+', x))).astype('uint')

# Difference between the item's price and it's category's median price
median_prices_per_cat = data.groupby('category_name')['price'].median()
data['category_price_diff'] = data['price'] - data['category_name'].map(median_prices_per_cat)

# Drop unneeded columns
cols_to_drop = ['title', 'description', 'activation_date', 'user_id',
                'region', 'city']
data.drop(cols_to_drop, axis='columns', inplace=True)

# Save the features
is_train = data['deal_probability'].notnull()
data[is_train].to_csv('features/train/vanilla.csv', index=False)
data[~is_train].to_csv('features/test/vanilla.csv', index=False)
