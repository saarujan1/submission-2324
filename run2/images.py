from random import choices

import numpy as np

from PIL import Image
from sklearn.preprocessing import normalize

def load(path):
    '''load an image from a file as a numpy array'''

    return np.array(Image.open(path))

def save(image, path, mode=None):
    '''save an image in the from of a numpy array to a file'''

    Image.fromarray(image, mode).save(path)

#def mean_center(sample):
#    '''subtract the mean from each value in the sample'''
#
#    return sample - np.mean(sample)

# TODO test this!
def mean_center(arr):
    ''' subtract the mean vector from each part of a 2D array'''

    mean_vector = np.mean(arr, axis=0)
    return arr - np.tile(mean_vector, (arr.shape[0], 1))

# TODO delete this if its not needed!
def normalise(sample, minimum=-1, maximum=1):
    '''normalise the sample to minimum and maximum'''

    sample_minimum = np.min(sample)
    sample_maximum = np.max(sample)
    
    # special case if all values are the same
    if sample_minimum == sample_maximum:
        return np.full(sample.shape, (minimum + maximum) / 2)
    
    unit_normal = (sample - sample_minimum) / (sample_maximum - sample_minimum)

    return (unit_normal * (maximum - minimum)) + minimum

def largest_multiple_less_than(val, mult):
    ''' return the largest multiple of mult less than val'''

    return val - (val % mult)

def extract_clusters(image, size=8, freq=4, frac=0.1):
    '''extract densely-sampled pixel patches from an image'''

    # try to avoid wrong-sized clusters
    max_y = largest_multiple_less_than(image.shape[0], size)
    max_x = largest_multiple_less_than(image.shape[1], size)

    # iterate over image every freq pixels in x and y
    samples = list()
    for y in range(0, max_y, freq):
        for x in range(0, max_y, freq):
            # extract a size-by-size sample
            sample = image[y : y + size, x : x + size]

            # filter out wrong-sized clusters
            if sample.shape != (size, size):
                continue

            samples.append(sample.flatten())

    # reduce the number of clusters
    if 0 < frac < 1:
        samples = choices(samples, k=int(len(samples) * frac))

    # pre-process the samples and return them as a 2D array
    sample_array = np.stack(samples, axis=0)
    return normalize(mean_center(sample_array))

