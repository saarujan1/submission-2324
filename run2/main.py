import sys
import pickle
import time

from argparse import ArgumentParser, FileType
from pathlib import Path
from operator import methodcaller
from multiprocessing import Pool

import numpy as np

import images
from classifier import Classifier

parser = ArgumentParser()
parser.add_argument('test_data', type=Path)

model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument('-t', '--training-data', nargs='+', type=Path, default=None)
model_group.add_argument('-m', '--model-cache', type=FileType('rb'), default=None)

parser.add_argument('-s', '--size', type=int, default=8)
parser.add_argument('-f', '--freq', type=int, default=4)

parser.add_argument('-o', '--outfile', type=FileType('w'), default=sys.stdout)
parser.add_argument('-l', '--logfile', type=FileType('w'), default=sys.stderr)

def load_images(directory, extension='jpg'):
    for path in directory.glob(f'*.{extension}'):
        yield images.load(path), path.name

def closest_category(image, classifiers):
    samples = images.extract_clusters(image)
    key = methodcaller('classify', samples)
    return min(classifiers, key=key).name

if __name__ == '__main__':
    args = parser.parse_args()

    if args.model_cache is not None:
        # load the cached model if available
        args.logfile.write(f'loading model from {args.model_cache.name}\n')
        classifiers = pickle.load(args.model_cache)
    else:
        # create classifiers for each category using training data
        args.logfile.write('generating classifiers...\n')

        # use a threadpool to speed things up
        def train(training_images, name):
            classifier = Classifier.from_images(name, training_images, args.size, args.freq)
            args.logfile.write(f'    {name} ({len(training_images)} images)\n')
            return classifier

        training_data = list()
        for path in args.training_data:
            training_images = list(image for image, _ in load_images(path))
            training_data.append((training_images, path.name))

        start_time = time.time()
        with Pool(None) as pool:
            classifiers = pool.starmap(train, training_data)
        end_time = time.time()

        args.logfile.write(f'finished in {end_time - start_time} seconds\n')

        # non-parallel version
        # classifiers = list()
        # for training_images, name in training_data:
        #     classifiers.append(train(training_images, name))

        # cache result
        try:
            cache_path = Path(input('finished, enter cache path: '))
            cache_path.touch()  # make sure the file exists

            pickle.dump(classifiers, cache_path.open(mode='wb'))
        except Exception as e:
            args.logfile.write(f'could not save model: {e}. Skipping...\n')

    # classifiy the test data and output to run2.txt
    args.logfile.write('classifying test images...\n')

    # use a threadpool to speed things up again
    def classify(args):
        image, name = args
        classification = closest_category(image, classifiers)
        return f'{name} {classification}\n'

    test_images = list(load_images(args.test_data))
    args.logfile.write(f'loaded {len(test_images)} images from {args.test_data.name}.\n')
    results = map(classify, test_images)

    # print the results in order
    def key(string):
        return int(string.split('.')[0])

    for result in sorted(results, key=key):
        args.outfile.write(result)

