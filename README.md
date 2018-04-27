# Kaggle Avito Demand Prediction

## Setup

:snake: Use Python 3.6

Assuming you're using [Anaconda](https://anaconda.org/anaconda/python), create a virtual environment and activate it.

```sh
>>> conda create -n avito python=3.6 anaconda
>>> source activate avito
```

Install the requirements.

```sh
>>> pip install -r requirements.txt
```

Open a Python REPL and download the necessary `nltk` corpora.

```python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
```

Add the [files provided by Kaggle](https://www.kaggle.com/c/avito-demand-prediction/data) in the `data` directory. **Don't unzip them**. The `data` directory should contain the following files:

```sh
periods_test.csv.zip
periods_train.csv.zip
sample_submission.csv.zip
test.csv.zip
test_active.csv.zip
test_jpg.zip
train.csv.zip
train_active.csv.zip
train_jpg.zip
```

## Usage

The feature extraction is done separately from the model training. The features are extracted and saved in the `features` directory. The predictions are saved in the `submissions` directory. All the scripts in the `scripts` directory assume you're running them from the root of this project (so you don't have to do `cd scripts`).

### Feature extraction

```sh
>>> python scripts/extract_vanilla_features.py
>>> python scripts/extract_n_pixels_per_image.py
>>> python scripts/extract_word_embeddings.py
```

### Model training

```sh
>>> python scripts/train_lgbm.py
```
