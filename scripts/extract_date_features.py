import re
import math
from datetime import datetime
import numpy as np
import pandas as pd


# IN :Series  String OUT : Series
# Stupid but challenging and fun to code
def fill_nan_date(data):
    median = math.floor(data['activation_date'].astype('int64').median())
    median = datetime.fromtimestamp(median // 1000000000)
    data['activation_date'] = data['activation_date'].fillna(median)
    return(data)

if __name__ == '__main__':

    df_train = pd.read_csv('data/periods_train.csv.zip')
    df_train['is_train'] = 1
    df_test = pd.read_csv('data/periods_test.csv.zip')
    df_test['is_train'] = 0
    data = pd.concat([df_train,df_test], axis = 0, ignore_index = True)

    # Ram economy
    del df_train
    del df_test

    # Check if nan is present in item_id, date_from, date_to
    assert data['item_id'].isnull().sum().sum() == 0
    assert data['date_from'].isnull().sum().sum() == 0
    assert data['date_to'].isnull().sum().sum() == 0

    # Convert to datetime type
    data['date_from'] = pd.to_datetime(data['date_from'])
    data['date_to'] = pd.to_datetime(data['date_to'])
    data['activation_date'] = pd.to_datetime(data['activation_date'])

    data = fill_nan_date(data) # fill nan

    # Check if nan is present in activation_date
    assert data['activation_date'].isnull().sum().sum() == 0

    # Get year,month,day and dayofweek for activation_date, date_from, date_to
    col_date = ['activation_date','date_from','date_to']

    for col in col_date:
        data[col+'_month'] = data[col].dt.month
        data[col+'_year'] = data[col].dt.year
        data[col+'_day'] = data[col].dt.day
        data[col+'_dayofweek'] = data[col].dt.dayofweek

    # total of day the advertising was placed
    data['displayed_date'] = (data['date_to'] - data['date_from']).dt.total_seconds() / (24 * 60 * 60)

    assert data['displayed_date'].isnull().sum().sum() == 0
    data = data.drop(['activation_date','date_from','date_to'],axis = 1)

    train = data[data['is_train'] == 1].drop(['is_train'],axis = 1)
    train.to_csv('features/train/date.csv',index = False)
    del train
    test = data[data['is_train'] == 0].drop(['is_train'],axis = 1)
    test.to_csv('features/test/date.csv', index = False)
