import numpy as np
import pandas as pd
from tqdm import tqdm


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[
        target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / \
        (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * \
        (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(
            columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(
            columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


if __name__ == '__main__':

    train = pd.read_csv('data/train.csv.zip')
    test = pd.read_csv('data/test.csv.zip')

    trn, sub = target_encode(train['category_name'],
                             test['category_name'],
                             target=train['price'],
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)

    cats = ['user_type', 'param_1', 'param_2', 'param_3',
            'category_name', 'parent_category_name', 'region']
    quants = ['image_top_1', 'item_seq_number', 'price', 'deal_probability']

    for cat in tqdm(cats):
        for quant in quants:
            feature_name = '{}_{}_smoothing_mean'.format(cat, quant)
            trn, sub = target_encode(train[cat],
                                     test[cat],
                                     target=train[quant],
                                     min_samples_leaf=100,
                                     smoothing=10,
                                     noise_level=0.01)
            trn.name = feature_name
            sub_name = feature_name
            train = pd.concat([train, trn], axis=1)
            test = pd.concat([test, sub], axis=1)
    col_to_drop = ['user_id', 'region', 'city', 'parent_category_name', 'param_1',
                   'param_2', 'param_3', 'title', 'description', 'price',
                   'item_seq_number', 'activation_date', 'user_type',
                   'image', 'image_top_1', 'category_name']

    train.drop(col_to_drop + ['deal_probability'], axis=1, inplace=True)
    test.drop(col_to_drop, axis=1, inplace=True)

    train.to_csv('features/train/smooth_mean.csv', index=False)
    test.to_csv('features/test/smooth_mean.csv', index=False)
