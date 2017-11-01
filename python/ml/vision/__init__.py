import mxnet as mx
import numpy as np
from skimage import io, transform


def PreprocessImage(img):
    if img.shape[2] == 3:
        img = mx.nd.transpose(img, (2, 0, 1))

    img[0, :] -= 123.68
    img[1, :] -= 116.779
    img[2, :] -= 103.939

    return mx.nd.expand_dims(img, axis = 0)

def PostprocessImage(img):
    image = img[0].copy()
    image[0, :] += 123.68
    image[1, :] += 116.779
    image[2, :] += 103.939

    image = mx.nd.transpose(image, (1, 2, 0))
    image = mx.nd.clip(image, 0, 255).astype(np.uint8)

    return image


