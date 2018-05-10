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
>>> python scripts/features/active.py
>>> python scripts/features/image_quality.py
>>> python scripts/features/text_likelihoods.py
>>> python scripts/features/vanilla.py

>>> python scripts/models/determine_cv.py
>>> python scripts/models/lgbm.py
```

### Model training

```sh
>>> python scripts/train_lgbm.py
```

## Explanation

### Overview

TODO

### Features

All the scripts for extracting features are located in `scripts/features`.

`active.py`

Extracts basic features from `train_active.csv` and `test_active.csv`. For example for each `user_id` in `train.csv` (respectively `test.csv`) we check if it is contained in `train_active.csv` (respectively `test_active.csv`).

`image_quality.py`

For each image we compute:

- the number of pixels
- the sharpness
- the brightness
- the contrast

The general idea being that we want to get a rough idea of the quality of each image.

`imagenet.py`

We used the keras version of the pre-trained model [Xception](https://github.com/keras-team/keras/blob/master/keras/applications/xception.py) to recognize objects on images. Xception take as input an image and return labels of objects with the probability associated. 

We have some hypotheses about the output probabilities of the algorithm.

High probability :

- Common objects
- Objects easy to recognize 
- Same kind of objects in the Imagenet data set 

Low probability :

- Rare objects 
- Low quality of the picture
- Objects which are not in the Imagenet data set 

Pre-trained model allow us to save time, someone else has already spent the time to learn a lot of features. 

`text_embeddings.py`

TODO

`text_likelihoods.py`

This is a cool one :v:. The idea is that we want to determine how each word in the titles (and the descriptions) influence the deal probability. To do so we first calculate the average deal probability (we call it a posterior) per word (we actually clean and stem each word). Then, for each title (and description) we average the posterior of each token it contains. To counter the bias that occurs when tokens don't appear frequently we bias the posteriors like it is done [here](https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators).

`text_svd.py`

Here we shamelessly copied an idea from [SRK's notebook](https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-notebook-avito). We separately compute the TF-IDF matrix for the titles and the descriptions followed by a PCA to extract the main principal components.

`vanilla.py`

This extracts a bunch of basic features (which we call "vanilla"). The script is quite explicit and commented.

### Models

TODO
