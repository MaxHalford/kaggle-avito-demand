import collections
import string

from nltk import corpus
from nltk import stem
from nltk import tokenize
import pandas as pd
from tqdm import tqdm


STOP_WORDS = set(corpus.stopwords.words('russian'))
STOP_WORDS.update(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
STEMMER = stem.snowball.SnowballStemmer('russian')
TOKENIZER = tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+')
PUNCTUATION = str.maketrans({p: None for p in string.punctuation})
PRIOR = 0.14
PRIOR_WEIGHT = 100
COLUMN = 'title'


def tokenize(sentence):

    # Remove punctuation and lowercase
    title = sentence.translate(PUNCTUATION).lower()
    # Tokenize
    tokens = TOKENIZER.tokenize(title)
    # Remove stop words
    tokens = set(tokens).difference(STOP_WORDS)
    # Step tokens
    tokens = [STEMMER.stem(token) for token in tokens]

    return tokens


def mean(floats):
    return sum(floats) / len(floats)


if __name__ == '__main__':

    # Load data
    cols = ['item_id', COLUMN]
    train = pd.read_csv('data/train.csv.zip', usecols=cols + ['deal_probability'])
    test = pd.read_csv('data/test.csv.zip', usecols=cols)

    # Create lookup tables
    train_item_tokens = collections.defaultdict(list)
    test_item_tokens = collections.defaultdict(list)
    token_deal_probas = collections.defaultdict(list)

    # Loop over train rows
    for _, row in tqdm(train.iterrows()):
        for token in tokenize(row[COLUMN]):
            train_item_tokens[row['item_id']].append(token)
            token_deal_probas[token].append(row['deal_probability'])

    # Compute Bayesian posteriors
    # https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators
    token_deal_proba_means = {
        token: (mean(probas) * len(probas) + PRIOR * PRIOR_WEIGHT) / (len(probas) + PRIOR_WEIGHT)
        for token, probas in token_deal_probas.items()
    }

    # Loop over test rows
    for _, row in tqdm(test.iterrows()):
        for token in tokenize(row[COLUMN]):
            test_item_tokens[row['item_id']].append(token)

    # Compute average deal probabilities of each item's tokens
    feature_name = '{}_avg_deal_proba'.format(COLUMN)
    train[feature_name] = train['item_id'].map(
        lambda x: mean([token_deal_proba_means[token] for token in train_item_tokens[x]] or [PRIOR])
    )
    test[feature_name] = test['item_id'].map(
        lambda x: mean([token_deal_proba_means.get(token, PRIOR) for token in test_item_tokens[x]] or [PRIOR])
    )

    # Save features
    filename = '{}.csv'.format(feature_name)
    train[['item_id', feature_name]].to_csv('features/train/{}'.format(filename), index=False)
    test[['item_id', feature_name]].to_csv('features/test/{}'.format(filename), index=False)