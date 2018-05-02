import string

import gensim
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd


STOP_WORDS = set(
    stopwords.words('russian') +
    ['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на']
)

PUNCTUATION = str.maketrans({p: None for p in string.punctuation})


class DocumentIterator():

    def __init__(self, documents, names):
        self.documents = documents
        self.names = names

    def __iter__(self):
        for doc, name in zip(self.documents, self.names):
            yield gensim.models.doc2vec.TaggedDocument(doc, [name])


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
            pd.read_csv('data/train.csv.zip', usecols=columns + ['deal_probability']),
            pd.read_csv('data/test.csv.zip', usecols=columns)
        ),
        ignore_index=True
    )

    # Determine which rows are train ones
    train_mask = data['deal_probability'].notnull()

    # Description embeddings

    # Train
    data['description'].fillna('', inplace=True)
    tokens = data['description'].map(tokenize_ru)
    iterator = DocumentIterator(tokens, data['item_id'])
    model = gensim.models.Doc2Vec(min_count=5, vector_size=256)
    model.build_vocab(iterator)
    model.train(iterator, total_examples=model.corpus_count, epochs=1)
    docvecs = model.docvecs
    del model  # Saves up a lot of RAM

    # Save
    embeddings = pd.DataFrame(np.array([docvecs[name] for name in data['item_id']]))
    embeddings['item_id'] = data['item_id']
    assert len(embeddings) == len(data)
    embeddings[train_mask].to_csv('features/train/description_embeddings.csv', index=False)
    embeddings[~train_mask].to_csv('features/test/description_embeddings.csv', index=False)

    # Title embeddings

    # Train
    data['title'].fillna('', inplace=True)
    tokens = data['title'].map(tokenize_ru)
    iterator = DocumentIterator(tokens, data['item_id'])
    model = gensim.models.Doc2Vec(min_count=5, vector_size=256)
    model.build_vocab(iterator)
    model.train(iterator, total_examples=model.corpus_count, epochs=1)
    docvecs = model.docvecs
    del model  # Saves up a lot of RAM

    # Save
    embeddings = pd.DataFrame(np.array([docvecs[name] for name in data['item_id']]))
    embeddings['item_id'] = data['item_id']
    assert len(embeddings) == len(data)
    embeddings[train_mask].to_csv('features/train/title_embeddings.csv', index=False)
    embeddings[~train_mask].to_csv('features/test/title_embeddings.csv', index=False)
