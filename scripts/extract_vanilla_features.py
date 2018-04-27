import re

import numpy as np
import pandas as pd


data = pd.concat(
    (
        pd.read_csv('data/train.csv.zip'),
        pd.read_csv('data/test.csv.zip')
    ),
    ignore_index=True
)

# Number of adds per user
data = data.join(data.groupby('user_id').size().rename('user_n_ads'), on='user_id')

# Number of adds per city
data = data.join(data.groupby('city').size().rename('city_n_ads'), on='city')

# Number of adds per region
data = data.join(data.groupby('region').size().rename('region_n_ads'), on='region')

# Number of adds per parameter
data = data.join(data.groupby('param_1').size().rename('param_1_n_ads'), on='param_1')
data = data.join(data.groupby('param_2').size().rename('param_2_n_ads'), on='param_2')
data = data.join(data.groupby('param_3').size().rename('param_3_n_ads'), on='param_3')

# Image classification code
data['image_top_1'].fillna(-1, inplace=True)

# Is the price a round price (e.g. 100 or 1000, not 42 or 1337)
data['round_price'] = data['price'].map(
    lambda x: str(int(x)).endswith('0')
    if np.isfinite(x)
    else False
)

# Price
data['price'].fillna(-1, inplace=True)

# Number of characters in the description
data['description_n_characters'] = data['description'].fillna('').map(len)

# Number of words in the description
data['description_n_words'] = data['description'].fillna('').map(lambda x: len(re.findall(r'\w+', x)))

# Number of characters in the title
data['title_n_characters'] = data['title'].fillna('').map(len)

# Number of words in the title
data['title_n_words'] = data['title'].fillna('').map(lambda x: len(re.findall(r'\w+', x)))

# Drop unneeded columns
cols_to_drop = ['region', 'city', 'param_1', 'param_2', 'param_3',
                'title', 'description', 'activation_date', 'user_id']
data.drop(cols_to_drop, axis='columns', inplace=True)

# Save the features
is_train = data['deal_probability'].notnull()
data[is_train].to_csv('features/train/vanilla.csv', index=False)
data[~is_train].to_csv('features/test/vanilla.csv', index=False)
