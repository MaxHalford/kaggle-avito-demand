import pandas as pd


cols = ['user_id', 'item_id', 'title', 'city', 'region']
dtype = {col: 'category' for col in cols}

data = pd.concat(
    (
        pd.read_csv('data/train.csv.zip', usecols=cols + ['deal_probability'], dtype=dtype),
        pd.read_csv('data/test.csv.zip', usecols=cols, dtype=dtype)
    ),
    ignore_index=True
)

full = pd.merge(
    left=pd.concat(
        (
            data,
            pd.read_csv('data/train_active.csv.zip', usecols=cols, dtype=dtype).drop_duplicates('item_id'),
            pd.read_csv('data/test_active.csv.zip', usecols=cols, dtype=dtype).drop_duplicates('item_id')
        ),
        ignore_index=True
    ),
    right=pd.concat(
        (
            pd.read_csv('data/periods_train.csv.zip', parse_dates=['activation_date', 'date_from', 'date_to']),
            pd.read_csv('data/periods_test.csv.zip', parse_dates=['activation_date', 'date_from', 'date_to'])
        ),
        ignore_index=True
    ),
    how='left',
    on='item_id'
)

# Number of entries per user
data = data.join(full.groupby('user_id').size().rename('user_n_entries').astype('uint'), on='user_id')

# Number of items per user
data = data.join(full.groupby('user_id')['item_id'].nunique().rename('user_n_ads').astype('uint'), on='user_id')

# Number of adds per city
data['city'] = data['city'] + ', ' + data['region']
data = data.join(full.groupby('city').size().rename('city_n_entries').astype('uint'), on='city')

# Number of adds per region
data = data.join(full.groupby('region').size().rename('region_n_entries').astype('uint'), on='region')

# Number of adds per title
data = data.join(full.groupby('title').size().rename('title_n_entries').astype('uint'), on='title')

# Number of entries per ad
data = data.join(full.groupby(['user_id', 'title']).size().rename('ad_n_entries').astype('uint'), on=['user_id', 'title'])

# Average up time per user
full['days_up'] = (full['date_to'] - full['date_from']).dt.days
data = data.join(full.groupby('user_id')['days_up'].sum().rename('user_up_time').astype('uint'), on='user_id')

# Drop unneeded columns
cols_to_drop = ['user_id', 'title', 'city', 'region']
data.drop(cols_to_drop, axis='columns', inplace=True)

# Save features
train_mask = data['deal_probability'].notnull()
data[train_mask].drop('deal_probability', axis='columns').to_csv('features/train/active.csv', index=False)
data[~train_mask].drop('deal_probability', axis='columns').to_csv('features/test/active.csv', index=False)
