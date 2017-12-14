import numpy as np
import scipy.ndimage
import PIL.Image
from skimage import exposure


def main():
    filename = '2.jpeg'
    im = PIL.Image.open(filename)
    ima = np.asarray(im)
    img_eq = exposure.equalize_hist(ima)
    img_sobel = scipy.ndimage.sobel(img_eq, axis=0)
    im = PIL.Image.fromarray(img_sobel.astype('uint8'))
    im.save(filename.replace('.jpeg', '_sobeltest.png'))


if __name__ == '__main__':
    main()
