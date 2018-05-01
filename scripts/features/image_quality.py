import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from tqdm import tqdm


def image_sharpness(img):
    gray = img.convert('L')
    array = np.asarray(gray, dtype=np.int32)
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness


def image_brightness(img):
    gray = img.convert('L')
    histogram = gray.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


def image_contrast(img):
    histogram = img.convert('L').histogram()
    return stats.entropy(histogram)


def compute_pixels_per_image(path):

    archive = zipfile.ZipFile(path, 'r')
    files = [f for f in archive.namelist() if f.endswith('.jpg')]
    names = [os.path.basename(f).split('.')[0] for f in files]

    n_pixels = {}
    sharpness = {}
    brightness = {}
    contrast = {}
    n_errors = 0

    for f in tqdm(files):
        name = os.path.basename(f).split('.')[0]
        with archive.open(f) as file:
            try:
                img = Image.open(file)
                n_pixels[name] = img.size[0] * img.size[1]
                sharpness[name] = image_sharpness(img)
                brightness[name] = image_brightness(img)
                contrast[name] = image_contrast(img)
            except OSError:
                n_pixels[name] = None
                sharpness[name] = None
                brightness[name] = None
                contrast[name] = None
                n_errors += 1

    print('Number of errors: {}'.format(n_errors))

    features = pd.DataFrame.from_dict({
        name: [n_pixels.get(name), sharpness.get(name), brightness.get(name), contrast.get(name)]
        for name in names
    }, orient='index').reset_index()
    features.columns = ['image' 'n_pixels', 'sharpness', 'brightness', 'contrast']

    return features.fillna(features.mean())


if __name__ == '__main__':
    compute_pixels_per_image('data/train_jpg.zip').to_csv('features/train/image_features.csv', index=False)
    compute_pixels_per_image('data/test_jpg.zip').to_csv('features/test/image_features.csv', index=False)
