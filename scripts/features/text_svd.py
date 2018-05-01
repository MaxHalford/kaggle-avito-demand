from nltk.corpus import stopwords
import pandas as pd
from sklearn import decomposition
from sklearn import feature_extraction


data = pd.concat(
    (
        pd.read_csv('data/train.csv.zip'),
        pd.read_csv('data/test.csv.zip')
    ),
    ignore_index=True
)

train_mask = data['deal_probability'].notnull()

stop_words = set(stopwords.words('russian'))
stop_words.update(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
tfidf = feature_extraction.text.TfidfVectorizer(stop_words=stop_words)

# Title

tokens = tfidf.fit_transform(data['title'].fillna(''))

n_comp = 5
svd = decomposition.TruncatedSVD(n_components=n_comp, algorithm='arpack', n_iter=10)
components = pd.DataFrame(
    svd.fit_transform(tokens),
    columns=['title_comp_{}'.format(i+1) for i in range(n_comp)]
)
components['item_id'] = data['item_id'].values
components[train_mask].to_csv('features/train/title_svd.csv', index=False)
components[~train_mask].to_csv('features/test/title_svd.csv', index=False)

# Description

tokens = tfidf.fit_transform(data['description'].fillna(''))

n_comp = 5
svd = decomposition.TruncatedSVD(n_components=n_comp, algorithm='arpack', n_iter=10)
components = pd.DataFrame(
    svd.fit_transform(tokens),
    columns=['description_comp_{}'.format(i+1) for i in range(n_comp)]
)
components['item_id'] = data['item_id'].values
components[train_mask].to_csv('features/train/description_svd.csv', index=False)
components[~train_mask].to_csv('features/test/description_svd.csv', index=False)
