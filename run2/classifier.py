import numpy as np

from sklearn.cluster import KMeans

import images

class Classifier:
    '''one-vs-all classifier using vector quantisation'''

    def __init__(self, name, training_samples):
        self.name = name
        
        self._kmeans = KMeans(n_clusters=500, n_init='auto')
        self._kmeans.fit(training_samples)

    @classmethod
    def from_images(cls, name, training_images, size, freq):
        clusters = np.empty((0, size ** 2))
        for image in training_images:
            image_clusters = images.extract_clusters(
                image,
                size=size,
                freq=freq
            )
            clusters = np.concatenate((clusters, image_clusters))

        return cls(name, clusters)

    def classify(self, samples):
        '''returns how close the image is to this classifier's category'''
        
        return abs(self._kmeans.score(samples))

    def __call__(self, image):
        '''shortcut for Classifier.classify'''

        self.classify(image)

