import json
import string

import gensim
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras import models
from keras import layers
from keras import preprocessing
from keras import callbacks
import nltk
import numpy as np
import pandas as pd


COLUMN = 'title'  # description or title
TOKENIZER = nltk.tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+')
PUNCTUATION = str.maketrans({p: None for p in string.punctuation})
EMBEDDINGS_FILE = 'features/custom_embeddings{}.vec'.format(COLUMN)
PADDING = 10
N_MAX_WORDS = 100000
BATCH_SIZE = 256
NUM_EPOCHS = 8
NUM_FILTERS = 64
EMBEDDING_SIZE = 100
WEIGHT_DECAY = 1e-4


def clean_and_tokenize(text):

    # Clean
    text = text.translate(PUNCTUATION)\
               .replace('«', '')\
               .replace('»', '')\
               .lower()

    # Tokenize
    tokens = TOKENIZER.tokenize(text)

    return tokens


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


if __name__ == '__main__':

    # Load data
    columns = ['item_id', COLUMN]
    data = pd.concat(
        (
            pd.read_csv('data/train.csv.zip',
                        usecols=columns + ['deal_probability']),
            pd.read_csv('data/test.csv.zip', usecols=columns)
        ),
        ignore_index=True
    )

    # Load folds
    with open('folds/folds_item_ids.json') as infile:
        folds_item_ids = json.load(infile)

    # Load embeddings
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(
        EMBEDDINGS_FILE, binary=False)

    # Clean and tokenize
    tokens = data[COLUMN].fillna('').map(clean_and_tokenize).tolist()

    # Get token indexes
    idx_tokenizer = preprocessing.text.Tokenizer(num_words=N_MAX_WORDS)
    idx_tokenizer.fit_on_texts(tokens)
    tokens_idx = idx_tokenizer.texts_to_sequences(tokens)

    # Apply padding
    tokens_idx = preprocessing.sequence.pad_sequences(
        tokens_idx, maxlen=PADDING)

    # Build embedding matrix
    missing_tokens = []
    n_tokens = len(idx_tokenizer.word_index) + 1
    embedding_matrix = np.zeros((n_tokens, EMBEDDING_SIZE))
    for word, i in idx_tokenizer.word_index.items():
        try:
            embedding = embeddings.get_vector(word)
            embedding_matrix[i] = embedding
        except KeyError:
            missing_tokens.append(word)
    print('Number of words not found: {}'.format(len(missing_tokens)))

    # Initialize the model
    model = models.Sequential()
    model.add(layers.Embedding(
        n_tokens,
        EMBEDDING_SIZE,
        weights=[embedding_matrix],
        input_length=PADDING,
        trainable=False
    ))
    model.add(layers.Conv1D(NUM_FILTERS, 7, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(NUM_FILTERS, 7, activation='relu', padding='same'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu',
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
    model.add(layers.Dense(1))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9,
                           beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=adam,
                  metrics=[root_mean_squared_error])
    model.summary()
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=4,
        verbose=1
    )
    callbacks_list = [early_stopping]
    train_mask = data['deal_probability'].notnull()

    # Train
    for i in folds_item_ids.keys():

        fit_mask = data['item_id'].isin(folds_item_ids[i]['fit'])
        val_mask = data['item_id'].isin(folds_item_ids[i]['val'])

        hist = model.fit(
            tokens_idx[fit_mask],
            data['deal_probability'][fit_mask],
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            callbacks=callbacks_list,
            validation_split=0.1,
            shuffle=True,
            verbose=2
        )

        # Save out-of-fold and test predictions
        y_val = model.predict(tokens_idx[val_mask])
        pd.Series(y_val[:,  0]).to_csv(
            'folds/cnn_{}_val_{}.csv'.format(COLUMN, i), index=False)
        y_test = model.predict(tokens_idx[~train_mask])
        pd.Series(y_test[:,  0]).to_csv(
            'folds/cnn_{}_test_{}.csv'.format(COLUMN, i), index=False)
