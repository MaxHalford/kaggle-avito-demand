import pandas as pd


columns = ['region', 'city', 'item_id']
data = pd.concat(
    (
        pd.read_csv('data/train.csv.zip',
                    usecols=columns + ['deal_probability']),
        pd.read_csv('data/test.csv.zip',
                    usecols=columns)
    ),
    ignore_index=True
)

train_mask = data['deal_probability'].notnull()
data['location'] = data['city'] + ', ' + data['region']
locs_df = pd.read_csv('data/city_latlons.csv')
data = data.merge(locs_df, how='left', on='location')
del locs_df

common_city = data.location.mode()[0]  # Most frequent location

# geo_fill = [lat,lon]
geo_fill = data[['lat', 'lon']][data['location'] == common_city].values[0]
data['lat'].fillna(float(geo_fill[0]), inplace=True)
data['lon'].fillna(float(geo_fill[1]), inplace=True)

cols_to_drop = ['city', 'location', 'region', 'deal_probability']
data.drop(cols_to_drop, axis=1, inplace=True)

data[train_mask].to_csv(
    'features/train/geocode.csv', index=False)
data[~train_mask].to_csv(
    'features/test/geocode.csv', index=False)

