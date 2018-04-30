import pandas as pd


cols = ['user_id', 'item_id', 'title']
dtype = {
    'user_id': 'category',
    'item_id': 'category',
    'title': 'category'
}
train = pd.read_csv('data/train.csv.zip', usecols=cols, dtype=dtype)
test = pd.read_csv('data/test.csv.zip', usecols=cols, dtype=dtype)
train_active = pd.read_csv('data/train_active.csv.zip', usecols=cols, dtype=dtype)
test_active = pd.read_csv('data/test_active.csv.zip', usecols=cols, dtype=dtype)

print('Data is loaded (finally)')

train_features = train['item_id'].to_frame()
test_features = test['item_id'].to_frame()

# Is the user id in the supplementary data
train_features['user_id_in_sup'] = train['user_id'].isin(train_active['user_id'])
test_features['user_id_in_sup'] = test['user_id'].isin(test_active['user_id'])

# Is the title in the supplementary data
train_features['title_in_sup'] = train['title'].isin(train_active['title'])
test_features['title_in_sup'] = test['title'].isin(test_active['title'])

# Save features
train_features.to_csv('features/train/active.csv', index=False)
test_features.to_csv('features/test/active.csv', index=False)
