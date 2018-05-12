# https://www.kaggle.com/dhznsdl/nn-model-adding-variables-step-by-step
'''
# TODO:
ADD REGULARIZATION PARAMETERS (DROP_OUT, EarlyStopping)
ADD WORD EMBEDDING TITLE AND DESCRIPTION
'''
import numpy as np
import pandas as pd
import json

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import argparse
from keras.layers import Input, Embedding, Dense
from keras.layers import GlobalMaxPool1D, GlobalMaxPool2D
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Dropout
from keras.layers import LSTM, CuDNNGRU, CuDNNLSTM, GRU
from keras.optimizers import RMSprop, Adam
from keras import callbacks
from keras.preprocessing.sequence import pad_sequences


from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def tknzr_fit(col, trn, test):
    tknzr = Tokenizer(filters='', lower=False, split='뷁', oov_token='oov')
    tknzr.fit_on_texts(trn[col])
    return np.array(tknzr.texts_to_sequences(trn[col])), np.array(tknzr.texts_to_sequences(test[col])), tknzr


def get_model():
    K.clear_session()
    inp_reg = Input(shape=(1, ), name='inp_region')
    emb_reg = Embedding(config.len_reg, config.emb_reg,
                        name='emb_region')(inp_reg)

    inp_pcn = Input(shape=(1, ), name='inp_parent_category_name')
    emb_pcn = Embedding(config.len_pcn, config.emb_pcn,
                        name='emb_parent_category_name')(inp_pcn)

    inp_cn = Input(shape=(1, ), name='inp_category_name')
    emb_cn = Embedding(config.len_cn, config.emb_cn,
                       name="emb_category_name")(inp_cn)

    inp_ut = Input(shape=(1, ), name='inp_user_type')
    emb_ut = Embedding(config.len_ut, config.emb_ut,
                       name='emb_user_type')(inp_ut)

    inp_city = Input(shape=(1, ), name='inp_city')
    emb_city = Embedding(config.len_city, config.emb_city,
                         name='emb_city')(inp_city)

    inp_week = Input(shape=(1, ), name='inp_week')
    emb_week = Embedding(config.len_week, config.emb_week,
                         name='emb_week')(inp_week)

    inp_imgt1 = Input(shape=(1, ), name='inp_imgt1')
    emb_imgt1 = Embedding(config.len_imgt1, config.emb_imgt1,
                          name='emb_imgt1')(inp_imgt1)

    inp_p1 = Input(shape=(1, ), name='inp_p1')
    emb_p1 = Embedding(config.len_p1, config.emb_p1, name='emb_p1')(inp_p1)

    inp_p2 = Input(shape=(1, ), name='inp_p2')
    emb_p2 = Embedding(config.len_p2, config.emb_p2, name='emb_p2')(inp_p2)

    inp_p3 = Input(shape=(1, ), name='inp_p3')
    emb_p3 = Embedding(config.len_p3, config.emb_p3, name='emb_p3')(inp_p3)

    conc_cate = concatenate([emb_reg, emb_pcn,  emb_cn, emb_ut, emb_city, emb_week,
                             emb_imgt1, emb_p1, emb_p2, emb_p3], axis=-1, name='concat_categorcal_vars')
    conc_cate = GlobalMaxPool1D()(conc_cate)

    inp_price = Input(shape=(1, ), name='inp_price')
    emb_price = Dense(config.emb_price, activation='tanh',
                      name='emb_price')(inp_price)

    inp_itemseq = Input(shape=(1, ), name='inp_itemseq')
    emb_itemseq = Dense(config.emb_itemseq, activation='tanh',
                        name='emb_itemseq')(inp_itemseq)

    conc_cont = concatenate([conc_cate, emb_price, emb_itemseq], axis=-1)
    x = Dense(200, activation='relu')(conc_cont)
    x = Dense(50, activation='relu')(x)

    # text
    inp_desc = Input(shape=(config.maxlen, ), name='inp_desc')
    emb_desc = Embedding(config.len_desc, config.emb_desc,
                         name='emb_desc')(inp_desc)

    desc_layer = GRU(40, return_sequences=False)(emb_desc)

    conc_desc = concatenate([x, desc_layer], axis=-1)
    ###

    outp = Dense(1, activation='sigmoid', name='output')(conc_desc)

    model = Model(inputs=[inp_reg, inp_pcn, inp_cn, inp_ut, inp_city, inp_week, inp_imgt1, inp_p1, inp_p2, inp_p3,
                          inp_price, inp_itemseq, inp_desc], outputs=outp)
    return model


if __name__ == '__main__':

    train = pd.read_csv('data/train.csv.zip')
    x_test = pd.read_csv('data/test.csv.zip')

    y_train = train['deal_probability']
    x_train = train.drop(['deal_probability'], axis='columns')

    x_train['image_top_1'].fillna(value=3067, inplace=True)
    x_test['image_top_1'].fillna(value=3067, inplace=True)

    x_train['param_1'].fillna(value='_NA_', inplace=True)
    x_test['param_1'].fillna(value='_NA_', inplace=True)

    x_train['param_2'].fillna(value='_NA_', inplace=True)
    x_test['param_2'].fillna(value='_NA_', inplace=True)

    x_train['param_3'].fillna(value='_NA_', inplace=True)
    x_test['param_3'].fillna(value='_NA_', inplace=True)

    x_train['description'].fillna(value='_NA_', inplace=True)
    x_test['description'].fillna(value='_NA_', inplace=True)

    # create config init
    config = argparse.Namespace()

    reg_train, reg_test, reg_tknzr = tknzr_fit('region', x_train, x_test)
    pcn_train, pcn_test, pcn_tknzr = tknzr_fit(
        'parent_category_name', x_train, x_test)
    cn_train, cn_test, cn_tknzr = tknzr_fit('category_name', x_train, x_test)
    ut_train, ut_test, ut_tknzr = tknzr_fit('user_type', x_train, x_test)
    city_train, city_test, city_tknzr = tknzr_fit('city', x_train, x_test)

    tr_p1, te_p1, tknzr_p1 = tknzr_fit('param_1', x_train, x_test)
    tr_p2, te_p2, tknzr_p2 = tknzr_fit('param_2', x_train, x_test)
    tr_p3, te_p3, tknzr_p3 = tknzr_fit('param_3', x_train, x_test)

    week_train = pd.to_datetime(x_train['activation_date']
                                ).dt.weekday.astype(np.int32).values
    week_test = pd.to_datetime(
        x_test['activation_date']).dt.weekday.astype(np.int32).values

    week_train = np.expand_dims(week_train, axis=-1)
    week_test = np.expand_dims(week_test, axis=-1)

    imgt1_train = x_train['image_top_1'].astype(np.int32).values
    imgt1_test = x_test['image_top_1'].astype(np.int32).values

    imgt1_train = np.expand_dims(imgt1_train, axis=-1)
    imgt1_test = np.expand_dims(imgt1_test, axis=-1)

    eps = 1e-10
    price_train = np.log(x_train['price'] + eps)
    price_test = np.log(x_test['price'] + eps)
    price_train[price_train.isna()] = -1.
    price_test[price_test.isna()] = -1.

    price_train = np.expand_dims(price_train, axis=-1)
    price_test = np.expand_dims(price_test, axis=-1)

    tr_itemseq = np.log(x_train['item_seq_number'])
    te_itemseq = np.log(x_test['item_seq_number'])

    tr_itemseq = np.expand_dims(tr_itemseq, axis=-1)
    te_itemseq = np.expand_dims(te_itemseq, axis=-1)

    config.len_desc = 100000

    tknzr_desc = Tokenizer(num_words=config.len_desc, lower='True')
    tknzr_desc.fit_on_texts(x_train['description'].values)

    tr_desc_seq = tknzr_desc.texts_to_sequences(x_train['description'].values)
    te_desc_seq = tknzr_desc.texts_to_sequences(x_test['description'].values)

    config.maxlen = 75

    tr_desc_pad = pad_sequences(tr_desc_seq, maxlen=config.maxlen)
    te_desc_pad = pad_sequences(te_desc_seq, maxlen=config.maxlen)
    # categorical
    config.len_reg = len(reg_tknzr.word_index) + 1
    config.len_pcn = len(pcn_tknzr.word_index) + 1
    config.len_cn = len(cn_tknzr.word_index) + 1
    config.len_ut = len(ut_tknzr.word_index) + 1
    config.len_city = len(city_tknzr.word_index) + 1
    config.len_week = 7
    config.len_imgt1 = int(x_train['image_top_1'].max()) + 1
    config.len_p1 = len(tknzr_p1.word_index) + 1
    config.len_p2 = len(tknzr_p2.word_index) + 1
    config.len_p3 = len(tknzr_p3.word_index) + 1

    # continuous
    config.len_price = 1
    config.len_itemseq = 1

    # categorical
    config.emb_reg = 8
    config.emb_pcn = 4
    config.emb_cn = 8
    config.emb_ut = 2
    config.emb_city = 16
    config.emb_week = 4
    config.emb_imgt1 = 16
    config.emb_p1 = 8
    config.emb_p2 = 16
    config.emb_p3 = 16

    # continuous
    config.emb_price = 16
    config.emb_itemseq = 16

# text
    config.emb_desc = 100

    config.batch_size = 1024

    model = get_model()
    model.compile(optimizer=RMSprop(lr=0.0005, decay=0.00001),
                  loss=root_mean_squared_error, metrics=['mse', root_mean_squared_error])
    model.summary()

    X = np.array([reg_train, pcn_train, cn_train, ut_train, city_train,
                  week_train, imgt1_train, tr_p1, tr_p2, tr_p3, price_train, tr_itemseq])

    X_test = np.array([reg_test, pcn_test, cn_test, ut_test,
                       city_test, week_test, imgt1_test, te_p1, te_p2, te_p3, price_test, te_itemseq])

    X_test.append(te_desc_pad)
    Y = y_train

    early_stopping = callbacks.EarlyStopping(
        patience=6,
        mode='min'
    )

    callbacks_list = [early_stopping]

    with open('folds/folds_item_ids.json') as infile:
        folds_item_ids = json.load(infile)

    for i in folds_item_ids.keys():

        fit_mask = train['item_id'].isin(folds_item_ids[i]['fit'])
        fit_mask = np.array(fit_mask[fit_mask == True].index)
        val_mask = train['item_id'].isin(folds_item_ids[i]['val'])
        val_mask = np.array(val_mask[val_mask == True].index)

        X_fit = [x[fit_mask] for x in X]

        X_valid = [x[val_mask] for x in X]
        X_test = [x for x in X_test]

        Y_train = Y[fit_mask]
        Y_valid = Y[val_mask]
        X_fit.append(tr_desc_pad[fit_mask])
        X_valid.append(tr_desc_pad[val_mask])


        model.fit(x=X_fit, y=np.array(Y_train), validation_data=(
            X_valid, Y_valid), batch_size=config.batch_size, epochs=10, callbacks=callbacks_list)

    # Save out-of-fold and test predictions
        y_val = model.predict(X_valid)
        pd.Series(y_val[:,  0]).to_csv(
            'folds/NN_val_{}.csv'.format(i), index=False)
        y_test = model.predict(X_test)
        pd.Series(y_test[:,  0]).to_csv(
            'folds/NN_test_{}.csv'.format(i), index=False)
