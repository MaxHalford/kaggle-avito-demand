import os
import zipfile
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from cv2 import cvtColor, Laplacian, COLOR_BGR2GRAY, CV_64F

def export_features(image_property):
    df = pd.DataFrame.from_dict(image_property, orient='index')
    df.index.names = ['image']
    df = df.reset_index()
    return df

def image_blurness(pixels):
    pixels = cvtColor(pixels, COLOR_BGR2GRAY)
    fm = Laplacian(pixels, CV_64F).var()
    return fm

def compute_pixels_per_image(read_path, write_path):

    archive = zipfile.ZipFile(read_path, 'r')
    files = [f for f in archive.namelist() if f.endswith('.jpg')][:10]
    names = [os.path.basename(f).split('.')[0] for f in files]

    image_property = {}
    n_errors = 0
    compteur = 0
    
    for f in tqdm(files):

        compteur += 1

        if compteur == 5000 : 
            export_features(image_property).to_csv(write_path, index=False)
            compteur = 0

        name = os.path.basename(f).split('.')[0]
        
        with archive.open(f) as file:
        
            try:

                img = Image.open(file)
                pixels = np.array(img)

                blurness = image_blurness(pixels)

                image_property[name] = {
                    'blurness' : blurness
                }

            except:

                    n_errors += 1

                    image_property[name] = {
                        'blurness' : None
                    }

    print('Number of errors: {}'.format(n_errors))
                
    df = export_features(image_property)

    return df 

if __name__ == '__main__':

    read_path = 'data/train_jpg.zip'
    write_path = 'features/train/blurness.csv'

    compute_pixels_per_image(read_path, write_path).to_csv(write_path, index=False)

    read_path = 'data/test_jpg.zip'
    write_path = 'features/test/blurness.csv'

    compute_pixels_per_image(read_path, write_path).to_csv(write_path, index=False)