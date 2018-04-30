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
	P = imagenet_utils.decode_predictions(preds)
	return P

def export_features(imagenet):
	
	df = pd.DataFrame.from_dict(imagenet, orient='index')
	df.index.names = ['image']
	df = df.reset_index()

	# We do not replace nan values in columns which is type of string 
	# object_0, object_1, object_2, object_3, object_4
	df['mean_probability'].fillna(df['mean_probability'].mean(), inplace=True)
	df['std_probability'].fillna(df['std_probability'].mean(), inplace=True)
	df['object_0_probability'].fillna(df['object_0_probability'].mean(), inplace=True)
	df['object_1_probability'].fillna(df['object_1_probability'].mean(), inplace=True)
	df['object_2_probability'].fillna(df['object_2_probability'].mean(), inplace=True)
	df['object_3_probability'].fillna(df['object_3_probability'].mean(), inplace=True)
	df['object_4_probability'].fillna(df['object_4_probability'].mean(), inplace=True)

	return df


def extract_probability(files, input_shape, preprocess, model) : 

	imagenet = {}
	n_errors = 0
	compteur = 0
	
	for f in tqdm(files):
		
		name = os.path.basename(f).split('.')[0]
		
		with archive.open(f) as file:

			compteur +=1

			if compteur == 100000 : 
				export_features(imagenet).to_csv('features/train/imagenet.csv', index=False)
				compteur = 0

			try:

				image = load_img(file, target_size=input_shape)
				image = img_to_array(image)
				image = np.expand_dims(image, axis=0)
				image = preprocess(image)
				P = predict_label(model, image)
				probability = np.array([x[2] for x in P[0]])


				imagenet[name] = {
				'mean_probability' : probability.mean(),
				'std_probability' : probability.std(),
				}
				
				for i in range(len(P[0])):
					imagenet[name]['object_{}'.format(i)] = P[0][i][1] 
					imagenet[name]['object_{}_probability'.format(i)] =  P[0][i][2]

			except : 

				n_errors += 1

				imagenet[name] = {
				'mean_probability' : None,
				'std_probability' : None,
				'object_0': None,
				'object_0_probability': None,
				'object_1': None,
				'object_1_probability': None,
				'object_2': None,
				'object_2_probability': None,
				'object_3': None,
				'object_3_probability': None,
				'object_4': None,
				'object_4_probability': None,
				}



	print('Number of errors: {}'.format(n_errors))

	df = export_features(imagenet)

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

	extract_probability(files, input_shape, preprocess,  model)