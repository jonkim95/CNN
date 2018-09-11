import numpy as np
import os
import pickle
import tarfile
import urllib

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_DIR = '/tmp/cifar10_data'
NUM_CHANNELS = 3
NUM_CLASSES = 10 # Number of output classes
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

def get_num_channels(): return NUM_CHANNELS
def get_num_classes(): return NUM_CLASSES
def get_image_height(): return IMAGE_HEIGHT
def get_image_width(): return IMAGE_WIDTH

def download_and_extract_data():
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    print('File path : ' + filepath)
    if not os.path.exists(filepath):
        print('Beginning download from ' + DATA_URL)
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded' + filename + ', ' + str(statinfo.st_size), 'bytes.')
    if not os.path.exists(os.path.join(dest_directory, 'cifar-10-batches-py/')):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    print("Data stored in " + os.path.join(dest_directory, 'cifar-10-batches-py/'))


def get_data():
    download_and_extract_data()
    data_dicts = []
    for i in xrange(1, 7):
        filename = os.path.join(DATA_DIR, 'cifar-10-batches-py/data_batch_%d' % i)
        if i == 6: filename = os.path.join(DATA_DIR, 'cifar-10-batches-py/test_batch')
        if os.path.isfile(filename):
            print("Attempting to read from file : " + filename)
            with open(filename, 'rb') as fo:
                data_dict = pickle.load(fo)
                for img_num in xrange(len(data_dict['data'])):
                    image = rearrange_image(data_dict['data'][img_num])
                    data_dicts += [{'image': image, 'label': data_dict['labels'][img_num]}]
        else: 
            raise ValueError('Failed to find file: ' + filename)
    return data_dicts


# Rearranges a 1 * 3072 image array into a 32 * 32 * 3 3-D matrix
def rearrange_image(image):
    return np.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))