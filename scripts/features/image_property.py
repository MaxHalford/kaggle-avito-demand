import os
import zipfile
import numpy as np
import pandas as pd
import operator

from cv2 import *
from PIL import Image
from scipy import stats
from tqdm import tqdm
from collections import defaultdict
from skimage import feature
# To Do : add requirements scikit-image & opencv-python

def export_features(image_property):
    df = pd.DataFrame.from_dict(image_property, orient='index')
    df.index.names = ['image']
    df = df.reset_index()
    return df

def clustering_color(img):
    pixels = np.float32(img)
    # number of clusters required at end
    n_clusters = 5
    # cv2.TERM_CRITERIA_EPS : stop the algorithm iteration if specified accuracy is reached
    # cv2.TERM_CRITERIA_MAX_ITER : stop the algorithm after the specified number of iterations
    # the sum of both allow to stop the iteration when any of the above condition is met
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    # initial centers are choosen randomly : 
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centroids = cv2.kmeans(pixels.reshape((-1, 3)), n_clusters, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    dominant_color = palette[np.argmax(stats.itemfreq(labels)[:, -1])]
    return [x for x in dominant_color]

def color_analysis(img):
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]   
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis(img):
    # cut the images into two halves as complete average may give bias results
    size = img.size
    halves = (size[0]/2, size[1]/2)

    try:

        light_percent1, dark_percent1 = color_analysis(img.crop((0, 0, size[0], halves[1])))
        light_percent2, dark_percent2 = color_analysis(img.crop((0, halves[1], size[0], size[1])))
    
    except Exception as e:
        
        return None, None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    return light_percent, dark_percent

def image_blurness(pixels):
    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(pixels, cv2.CV_64F).var()
    return fm

def average_pixel_width(img):
    pixels = np.asarray(img.convert(mode='L'))
    edges_sigma1 = feature.canny(pixels, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (img.size[0]*img.size[1]))
    return apw*100

def get_average_color(pixels):
    average_color = [pixels[:, :, i].mean() for i in range(pixels.shape[-1])]
    return [x for x in average_color]

def compute_pixels_per_image(path):

    archive = zipfile.ZipFile(path, 'r')
    files = [f for f in archive.namelist() if f.endswith('.jpg')]
    names = [os.path.basename(f).split('.')[0] for f in files]

    image_property = {}
    n_errors = 0
    compteur = 0
    
    for f in tqdm(files):

        compteur += 1

        if compteur == 5000 : 
            export_features(image_property).to_csv('features/train/image_property.csv', index=False)
            compteur = 0

        name = os.path.basename(f).split('.')[0]
        
        with archive.open(f) as file:
        
            try:

                img = Image.open(file)
                pixels = np.array(img)

                blurness = image_blurness(pixels)
                pixel_width = average_pixel_width(img)
                light_percent, dark_percent = perform_color_analysis(img)
                dominant_color_1, dominant_color_2, dominant_color_3 =  clustering_color(img)
                average_color_1, average_color_2, average_color_3 = get_average_color(pixels)

                image_property[name] = {
                    'dominant_color_1': dominant_color_1,
                    'dominant_color_2': dominant_color_2,
                    'dominant_color_3': dominant_color_3,
                    'light_percent': light_percent,
                    'dark_percent' : dark_percent,
                    'blurness' : blurness,
                    'pixel_width' : pixel_width,
                    'average_color_1' : average_color_1,
                    'average_color_2' : average_color_2,
                    'average_color_3' : average_color_3,
                }

            except:

                n_errors += 1

                image_property[name] = {
                    'dominant_color_1': None,
                    'dominant_color_2': None,
                    'dominant_color_3': None,
                    'light_percent': None,
                    'dark_percent' : None,
                    'blurness' : None,
                    'pixel_width' : None,
                    'average_color_1' : None,
                    'average_color_2' : None,
                    'average_color_3' : None,
                }

    print('Number of errors: {}'.format(n_errors))
                
    df = export_features(image_property)

    return df 

if __name__ == '__main__':

    compute_pixels_per_image(
        'data/train_jpg.zip').to_csv(
        'features/train/image_property.csv', index=False)
