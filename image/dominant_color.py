import PIL.Image
import numpy as np
import time
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn.cluster import KMeans


def dbsan_colour(ima):
    w, h, d = ima.shape
    X = np.reshape(ima, (w*h, d))
    min_samples = int(0.05*len(X))

    db = DBSCAN(eps=0.3, min_samples=50).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    unique_labels = set(labels)
    labels = labels.reshape((w, h))

    for label in list(unique_labels):
        if label != -1:
            mask = labels == label
            ima[mask][:,0] = ima[mask][:,0].mean()
            ima[mask][:,1] = ima[mask][:,1].mean()
            ima[mask][:,2] = ima[mask][:,2].mean()

    im = PIL.Image.fromarray(ima, 'RGB')
    return im


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
            print(cluster_int)
            colour_distribution[cluster_int] += 1
    for k, v in colour_distribution.items():
        v = v / (w*h)
        colour_distribution[k] = v
    im = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
    return im, colour_distribution

def main():
    filename = '1.png'
    im = PIL.Image.open(filename)
    ima = np.asarray(im)[:,:,:3]

    im = dbsan_colour(ima)
    print('saving now')
    filename1 = re.sub(r'(\.jpeg|\.jpg|\.png)', '_dbscan.png', filename)
    im.save(filename1)

    #im = kmeans_colour(ima)
    #filename2 = re.sub(r'(\.jpeg|\.jpg|\.png)', '_kmeans.png', filename)
    #im.save(filename2)


if __name__ == '__main__':
    main()
