import os
import zipfile

from tqdm import tqdm
import pandas as pd
from PIL import Image


def compute_pixels_per_image(path):

    archive = zipfile.ZipFile(path, 'r')
    files = [f for f in archive.namelist() if f.endswith('.jpg')]

    n_pixels = {}
    n_errors = 0

    for f in tqdm(files):
        name = os.path.basename(f).split('.')[0]
        with archive.open(f) as file:
            try:
                img = Image.open(file)
                n_pixels[name] = img.size[0] * img.size[1]
            except OSError:
                n_pixels[name] = None
                n_errors += 1

    print('Number of errors: {}'.format(n_errors))

    df = pd.Series(n_pixels).to_frame().reset_index()
    df.columns = ['image', 'n_pixels']
    df['n_pixels'].fillna(df['n_pixels'].mean(), inplace=True)
    return df


if __name__ == '__main__':
    compute_pixels_per_image('data/train_jpg.zip').to_csv('features/train/n_pixels.csv', index=False)
    compute_pixels_per_image('data/test_jpg.zip').to_csv('features/test/n_pixels.csv', index=False)
