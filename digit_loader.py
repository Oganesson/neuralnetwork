from imageio import imread, imwrite
from scipy import ndimage
import skimage
from skimage import color, transform, util
import numpy as np

def slice_size(x, y):
    return x.stop - x.start, y.stop - y.start

def slice_area(x, y):
    w, h = slice_size(x, y)
    return w * h

def slice_aspect_ratio(x, y):
    w, h = slice_size(x, y)
    return max(w / h, h / w)

def pad_square(im, fill):
    w, h = im.shape
    target = max(w, h)
    padw = max(0, h - w) // 2
    padh = max(0, w - h) // 2
    shape = ((padw, target - (w + padw)), (padh, (target - (h + padh))))
    return np.pad(im, shape, 'constant', constant_values=fill)

def pad_to(im, size, fill):
    w, h = im.shape
    tw, th = size
    padw = max(0, tw - w) // 2
    padh = max(0, th - h) // 2
    shape = ((padw, tw - (w + padw)), (padh, (th - (h + padh))))
    return np.pad(im, shape, 'constant', constant_values=fill)

def auto_digitize(path):
    # Convert the image to floating-point grayscale
    im = imread(path)
    im = skimage.color.rgb2gray(im)
    im = skimage.util.img_as_float(im)
    
    label_struct = ndimage.generate_binary_structure(2, 2)
    labels, _ = ndimage.label(im < 0.5, structure=label_struct)

    digit_ims = []
    for sx, sy in ndimage.find_objects(labels):
        # Discard objects smaller than 10*10 or with an aspect ratio
        # greater than 10.  These shapes are generally noise, when
        # considering scanned handwritten digits.
        if slice_area(sx, sy) < 10 * 10 or slice_aspect_ratio(sx, sy) > 10:
            continue
            
        # Extract the image slice and make it square
        di = im[sx, sy]
        di = pad_square(di, 1)

        # Scale the image to 20x20
        di = skimage.transform.resize(di, (20, 20), mode='constant')
        
        # Pad to 28x28.  Normally, this would be done about the center
        # of mass, but for my test cases the CoM was always within 1
        # pixel of the center, so this step is omitted
        di = pad_to(di, (28, 28), 1)

        digit_ims.append(di)

    return digit_ims