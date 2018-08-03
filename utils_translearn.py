#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-03 11:42:59
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang


import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.preprocessing import image
import h5py


def generate_grid_img(img_array, gap=0.1, background=1):

    """
    Assemble a matrix of images into a grid image
    :param img_array: input image matrix, use Python native list, each element
        in the list is a Numpy array.
    :param gap: portion of gap between images, relative ratio to image
        dimension
    :return: a single Numpy array
    """

    gap = float(gap)
    row_num = len(img_array)
    col_num = len(img_array[0])
    img_row_num, img_col_num, img_color_num = img_array[0][0].shape
    row_pixel_gap = int(img_row_num * gap)
    col_pixel_gap = int(img_col_num * gap)
    total_row_num = row_num * img_row_num + (row_num - 1) * row_pixel_gap
    total_col_num = col_num * img_col_num + (col_num - 1) * col_pixel_gap
    grid_img = (np.ones((total_row_num, total_col_num, img_color_num)) *
                background)

    for row_id in xrange(len(img_array)):
        for col_id in xrange(len(img_array[0])):
            row_index_start = row_id * (img_row_num + row_pixel_gap)
            row_index_end = row_index_start + img_row_num
            col_index_start = col_id * (img_col_num + col_pixel_gap)
            col_index_end = col_index_start + img_col_num
            grid_img[row_index_start:row_index_end,
                     col_index_start:col_index_end, ] = img_array[row_id][col_id]

    return grid_img


def dump_image(x, filename, format):

    img = image.array_to_img(x, scale=False)
    img.save(filename, format)

    return


def load_dataset(data_filename, keys=None):

    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename) as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))
    return dataset


def fix_gpu_memory(mem_fraction=1):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)

    return sess


def cal_rmsd(X, X_adv):

    rmsd = np.mean(np.square(X - X_adv), axis=tuple(range(1, X.ndim)))
    rmsd = np.sqrt(rmsd)
    avg_rmsd = np.mean(rmsd)
    std_rmsd = np.std(rmsd)

    return avg_rmsd, std_rmsd


def preprocess(X, method):

    # assume color last
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_preprocessing(X)
    elif method is 'inception':
        X = inception_preprocessing(X)
    elif method is 'mnist':
        X = mnist_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def reverse_preprocess(X, method):

    # assume color last
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_reverse_preprocessing(X)
    elif method is 'inception':
        X = inception_reverse_preprocessing(X)
    elif method is 'mnist':
        X = mnist_reverse_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def inception_reverse_preprocessing(x):

    x = np.array(x)

    x /= 2.0
    x += 0.5
    x *= 255.0

    return x


def inception_preprocessing(x):

    x = np.array(x)

    x /= 255.0
    x -= 0.5
    x *= 2.0

    return x


def mnist_preprocessing(x):

    x = np.array(x)
    x /= 255.0

    return x


def mnist_reverse_preprocessing(x):

    x = np.array(x)
    x *= 255.0

    return x


def imagenet_preprocessing(x, data_format=None):

    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    x = np.array(x)
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]
    std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]

    return x


def imagenet_reverse_preprocessing(x, data_format=None):

    """ Reverse preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """

    x = np.array(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    if data_format == 'channels_first':
        if x.ndim == 3:
            # Zero-center by mean pixel
            x[0, :, :] += 103.939
            x[1, :, :] += 116.779
            x[2, :, :] += 123.68
            # 'BGR'->'RGB'
            x = x[::-1, :, :]
        else:
            x[:, 0, :, :] += 103.939
            x[:, 1, :, :] += 116.779
            x[:, 2, :, :] += 123.68
            x = x[:, ::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[..., 0] += 103.939
        x[..., 1] += 116.779
        x[..., 2] += 123.68
        # 'BGR'->'RGB'
        x = x[..., ::-1]
    return x
