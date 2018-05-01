import pandas as pd
import json
import codecs

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras import models
from keras import layers
from keras import utils
from keras import preprocessing
from keras import callbacks
from nltk import corpus
from nltk import tokenize
import numpy as np
from tqdm import tqdm

# description or title
COLUMN = 'title'

data = pd.concat(
    (
        pd.read_csv('data/train.csv.zip').sample(400000),
        pd.read_csv('data/test.csv.zip')
    ),
    ignore_index=True
)

data[COLUMN].fillna('', inplace=True)

################################################################################



N_MAX_WORDS = 100000

stop_words = set(corpus.stopwords.words('russian'))
stop_words.update(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])


embeddings = {}
with codecs.open('data/cc.ru.300.vec', encoding='utf-8') as f:
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings[word] = embedding

################################################################################

tokenizer = tokenize.RegexpTokenizer(r'\w+')

doc_tokens = []
for doc in tqdm(data[COLUMN]):
    tokens = tokenizer.tokenize(doc)
    tokens = set(tokens).difference(stop_words)
    doc_tokens.append(list(tokens))

################################################################################

tokenizer = preprocessing.text.Tokenizer(num_words=N_MAX_WORDS, lower=True)
tokenizer.fit_on_texts(doc_tokens)
doc_tokens_idx = tokenizer.texts_to_sequences(doc_tokens)

################################################################################

doc_tokens_idx = preprocessing.sequence.pad_sequences(doc_tokens_idx, maxlen=30)

################################################################################

#training params
batch_size = 256
num_epochs = 8

#model parameters
num_filters = 64
embed_dim = 300
weight_decay = 1e-4

################################################################################

words_not_found = []
n_words = min(100000, len(tokenizer.word_index))
embedding_matrix = np.zeros((n_words, embed_dim))
for word, i in tokenizer.word_index.items():
    if i >= n_words:
        continue
    embedding = embeddings.get(word)
    if (embedding is not None) and len(embedding) > 0:
        embedding_matrix[i] = embedding
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

################################################################################

print("sample words not found: ", np.random.choice(words_not_found, 10))

################################################################################

model = models.Sequential()
model.add(layers.Embedding(n_words, embed_dim,
          weights=[embedding_matrix], input_length=30, trainable=False))
model.add(layers.Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Dense(1))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam, metrics=[root_mean_squared_error])
model.summary()

################################################################################

early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]
train_mask = data['deal_probability'].notnull()

# Load folds
with open('folds/folds_item_ids.json') as infile:
    folds_item_ids = json.load(infile)


for i in folds_item_ids.keys():

    # Determine train and val folds
    fit_mask = data['item_id'].isin(folds_item_ids[i]['fit'])
    val_mask = data['item_id'].isin(folds_item_ids[i]['val'])
    hist = model.fit(
        doc_tokens_idx[fit_mask],
        data['deal_probability'][fit_mask],
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=callbacks_list,
        validation_split=0.1,
        shuffle=True,
        verbose=2
        )

################################################################################

    y_val = model.predict(doc_tokens_idx[val_mask])
    pd.Series(y_val[:,  0]).to_csv('folds/cnn_{}_val_{}.csv'.format(COLUMN, i),index = False)
    y_test = model.predict(doc_tokens_idx[~train_mask])
    pd.Series(y_test[:,  0]).to_csv('folds/cnn_{}_test_{}.csv'.format(COLUMN, i),index = False)
