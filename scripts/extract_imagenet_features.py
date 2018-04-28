import numpy as np
import zipfile
from tqdm import tqdm
import pandas as pd
import os 

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


def initializate_model(model_name, MODELS):
	Network = MODELS[model_name]
	model = Network(weights="imagenet")
	return model 

def get_input_shape(model_name) : 
    if model_name in ("inception", "xception"):
        input_shape = (299, 299)
        preprocess = preprocess_input
    else : 
        input_shape = (224, 224)
        preprocess = imagenet_utils.preprocess_input
    return input_shape, preprocess

def predict_label(model, image): 
	preds = model.predict(image)
	probability = imagenet_utils.decode_predictions(preds)
	return probability


def extract_probability(files, input_shape, preprocess, model) : 

	imagenet = {}
	n_errors = 0
	
	for f in tqdm(files):
		
		name = os.path.basename(f).split('.')[0]
		
		with archive.open(f) as file:
			
			try:

				image = load_img(file, target_size=input_shape)
				image = img_to_array(image)
				image = np.expand_dims(image, axis=0)
				image = preprocess(image)
				probability = predict_label(model, image)
				probability = np.array([x[2] for x in probability[0]])


				imagenet[name] = {
				'mean_probability' : probability.mean(),
				'std_probability' : probability.std(),
				'median_probability' : probability[2],
				'max_probability' : probability[0]
				}

			except : 

				n_errors += 1

				imagenet[name] = {
				'mean_probability' : None,
				'std_probability' : None,
				'median_probability' : None,
				'max_probability' : None
				}


	df = pd.DataFrame.from_dict(imagenet, orient='index')
	df.index.names = ['image']
	df = df.reset_index()

	df['mean_probability'].fillna(df['mean_probability'].mean(), inplace=True)
	df['std_probability'].fillna(df['std_probability'].mean(), inplace=True)
	df['median_probability'].fillna(df['median_probability'].mean(), inplace=True)
	df['max_probability'].fillna(df['max_probability'].mean(), inplace=True)

	print('Number of errors: {}'.format(n_errors))

	return df

if __name__ == '__main__' : 

	path = 'data/train_jpg.zip'
	archive = zipfile.ZipFile(path, 'r')
	files = [f for f in archive.namelist() if f.endswith('.jpg')]
	
	MODELS = {
		"vgg16": VGG16,
		"vgg19": VGG19,
		"inception": InceptionV3,
		"xception": Xception, # TensorFlow ONLY
		"resnet": ResNet50
	}

	model_name = "inception"
	input_shape, preprocess = get_input_shape(model_name)
	model = initializate_model(model_name, MODELS)

	extract_probability(files, input_shape, preprocess,  model).to_csv('features/train/imagenet.csv', index=False)