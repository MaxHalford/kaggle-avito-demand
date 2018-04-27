import collections
import string

import gensim
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn import feature_extraction


STOP_WORDS = set(
    stopwords.words('russian') +
    ['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на']
)

PUNCTUATION = str.maketrans({p: None for p in string.punctuation})


class TfIdfEmbeddingVectorizer():

    def __init__(self, word2vec, size):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())
        self.size = size

    def fit(self, X):
        tfidf = feature_extraction.text.TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        maxidf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: maxidf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.size)], axis=0)
                for words in X
            ])


def tokenize_ru(text):
    """Source: http://zabaykin.ru/?tag=nltk"""

    # Delete punctuation symbols
    text = text.translate(PUNCTUATION)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Delete stop_words
    tokens = list(set(tokens).difference(STOP_WORDS))

    # Clean tokens
    tokens = [token.replace('«', '').replace('»', '').lower() for token in tokens]

    return tokens


if __name__ == '__main__':

    # Concatenate training and test data
    columns = ['description', 'title', 'item_id']
    data = pd.concat(
        (
            pd.read_csv('data/train.csv.zip', usecols=columns),
            pd.read_csv('data/test.csv.zip', usecols=columns)
        ),
        ignore_index=True
    )

    # Determine which rows are train ones
    train_mask = data['deal_probability'].notnull()

    # Extract description embeddings

    data['description'] = data['description'].fillna('')
    tokenized_description = data['description'].map(tokenize_ru).tolist()

    embedding_size = 300
    model_description = gensim.models.Word2Vec(tokenized_description, size=embedding_size)
    w2v_description = dict(zip(model_description.wv.index2word, model_description.wv.vectors))

    model = TfIdfEmbeddingVectorizer(w2v_description, embedding_size)
    model.fit(data['description'])
    description_embeddings = pd.DataFrame(model.transform(data['description']))

    print(len(description_embeddings))

    description_embeddings = pd.concat([description_embeddings, data[['item_id']]], axis='columns')
    description_embeddings[train_mask].to_csv('features/train/description_embeddings.csv', index=False)
    description_embeddings[~train_mask].to_csv('features/test/description_embeddings.csv', index=False)

    # Extract title embeddings

    # data['title'] = data['title'].fillna('')
    # tokenized_title = data['title'].map(tokenize_ru).tolist()
    # size_title = 300
    # model_title = gensim.models.Word2Vec(tokenized_title, size=size_title)
    # w2v_title = dict(zip(model_title.wv.index2word, model_title.wv.vectors))

    # word_vec_title = TfIdfEmbeddingVectorizer(w2v_title,size_title)

    # word_vec_title.fit(data['title'])
    # embedding_title = embedding.transform(data['title'])
    # embedding_title = pd.DataFrame(embedding_description)
    # embedding = pd.concat([embedding_description, embedding_title], axis=1)
