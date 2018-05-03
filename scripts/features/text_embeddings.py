import string

import gensim
import nltk
import pandas as pd


COLUMN = 'title'
TOKENIZER = nltk.tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+')
PUNCTUATION = str.maketrans({p: None for p in string.punctuation})
EMBEDDING_SIZE = 100
OUTPUT_FILE = 'features/custom_embeddings.vec'


def clean_and_tokenize(text):

    # Clean
    text = text.translate(PUNCTUATION)\
               .replace('«', '')\
               .replace('»', '')\
               .lower()

    # Tokenize
    tokens = TOKENIZER.tokenize(text)

    return tokens


if __name__ == '__main__':

    # Load data
    columns = ['item_id', COLUMN]
    data = pd.concat(
        (
            pd.read_csv('data/train.csv.zip', usecols=columns),
            pd.read_csv('data/train_active.csv.zip', usecols=columns),
            pd.read_csv('data/test.csv.zip', usecols=columns)
        ),
        ignore_index=True
    )

    # Prepare tokens
    tokens = data[COLUMN].fillna('').map(clean_and_tokenize).tolist()

    # Instantiate the model
    model = gensim.models.fasttext.FastText(
        size=EMBEDDING_SIZE,
        seed=42,
        min_count=5
    )

    # Train the model
    model.build_vocab(tokens)
    model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)

    # Save word vectors
    model.wv.save_word2vec_format(OUTPUT_FILE)
