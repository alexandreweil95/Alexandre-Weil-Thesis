# Data Augmentation operations

import random
import numpy as np
import cv2

def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val) # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res

def random_rotate_img(images):
    rand_roat = np.random.randint(4, size=1)
    angle = 90*rand_roat
    center = (images.shape[0] / 2, images.shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle[0], scale=1.0)

    img_inst = cv2.warpAffine(images, rot_matrix, dsize=images.shape[:2], borderMode=cv2.BORDER_CONSTANT)

    return img_inst

def random_crop(image, crop_size=(400, 400)):
    height, width = image.shape[:-1]
    dy, dx = crop_size
    X = np.copy(image)
    aX = np.zeros(tuple([3, 400, 400]))
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    aX = X[y:(y + dy), x:(x + dx), :]
    return aX


##### Additional Data Augmentation

# https://albumentations.readthedocs.io/en/latest/api/augmentations.html


from functools import wraps
from warnings import warn
from itertools import product
from scipy.ndimage.filters import gaussian_filter

import math
import random
import warnings
from enum import IntEnum
from types import LambdaType

import cv2
import numpy as np
# from skimage.measure import label

# import albumentations.augmentations.functional as F

# https://albumentations.readthedocs.io/en/latest/_modules/albumentations/augmentations/transforms.html#RandomBrightness

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def _brightness_contrast_adjust_non_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.dtype
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = np.dtype("uint8")

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = np.arange(0, max_value + 1).astype("float32")

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += beta * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img



contrast_limit=[-0.2,0.2]
brightness_limit=[-0.2,0.2]




def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):

    alpha_new = alpha + random.uniform(contrast_limit[0], contrast_limit[1])
    beta_new = beta + random.uniform(brightness_limit[0], brightness_limit[1])
    
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha_new, beta_new, beta_by_max)

    return _brightness_contrast_adjust_non_uint(img, alpha_new, beta_new, beta_by_max)



# Resizing

def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1

def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def resize_function(img, height, width, interpolation=cv2.INTER_LINEAR):
    img_height, img_width = img.shape[:2]
    if height == img_height and width == img_width:
        return img
    resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(width, height), interpolation=interpolation)
    return resize_fn(img)


def scale_function(img, scale, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    return resize_function(img, new_height, new_width, interpolation)


scale_limit = [0.9, 1.1]

def Ramdom_scaling(image, Chosen_scale=0):
        Chosen_scale = Chosen_scale + random.uniform(scale_limit[0], scale_limit[1])
        return scale_function(image, Chosen_scale)







