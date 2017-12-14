import time
import re
import os
from collections import defaultdict

import PIL.Image
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn.cluster import KMeans


def kmeans_colour(im):
    ima = np.asarray(im)[:,:,:3]
    w,h,d = ima.shape
    X = np.reshape(ima, (w*h, d))
    kcluster = cluster.KMeans(init='k-means++', n_clusters=10)
    kcluster.fit(X)
    predictions = kcluster.predict(X)
    clusters = kcluster.cluster_centers_
    image = np.zeros((w, h, 3))
    predictions_idx = 0
    colour_distribution = defaultdict(int)
    for i in range(w):
        for j in range(h):
            cluster_int = predictions[predictions_idx]
            image[i][j] = clusters[cluster_int]
            predictions_idx += 1
            colour_distribution[cluster_int] += 1
    for k, v in colour_distribution.items():
        v = v / (w*h)
        colour_distribution[k] = v
    im = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
    return im, colour_distribution


def main():
    for filename in os.listdir('data/resized-500/'):
        if filename.endswith(".png"):
            print(filename)
            im = PIL.Image.open('data/resized-500/%s' %filename)
            ima = np.asarray(im)[:,:,:3]
            im, dist = kmeans_colour(ima)
            im.save('data/kmeans_color-resized-500/%s' %filename)

if __name__ == '__main__':
    main()
