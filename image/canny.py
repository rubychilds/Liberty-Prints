import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature
import numpy as np
import PIL.Image

from skimage import data, img_as_float
from skimage import exposure


def test_canny(im):

    img_eq = exposure.equalize_hist(im)
    img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03)

    edges = []
    # Compute the Canny filter for two values of sigma
    edges.append([im, feature.canny(im), feature.canny(im, sigma=3)])
    edges.append([img_eq*255, feature.canny(img_eq), feature.canny(img_eq, sigma=2)])
    edges.append([img_adapteq*255, feature.canny(img_adapteq), feature.canny(img_adapteq, sigma=2)])

    for i, row in enumerate(edges):
        for j, edge in enumerate(row):
            edge = edge.astype('uint8')
            if edge.max() <= 1:
                edge = edge*255
            PIL.Image.fromarray(edge).save(filename.replace('.jpeg','_%d_%d.png'%(i,j)))
    return edges


def display(edges):
    fig, axis_ = plt.subplots(nrows=3, ncols=3, figsize=(8, 9), sharex=True, sharey=True)
    for i, row in enumerate(edges):
        for j, edge in enumerate(row):
            axis_[i][j].imshow(edge, cmap=plt.cm.gray)
    fig.tight_layout()
    plt.show()


def main():
    filename = '2.jpeg'
    img = PIL.Image.open(filename).convert('L')
    im = np.asarray(img)
    edges = test_canny(im)
    display(edges)


if __name__ == '__main__':
    main()
