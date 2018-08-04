from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf


def _FSpecialGauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    window = g / g.sum()
    window = np.reshape(window, (size, size, 1, 1))
    window = tf.constant(window, dtype=tf.float32)

    return window


def _Conv2dIndividual(input_tensor, filter_tensor, strides, padding):

    if input_tensor.get_shape().as_list()[3] == 3:
        input_split = tf.unstack(input_tensor, 3, axis=3)
        result_split = [tf.nn.conv2d(tf.expand_dims(in_t, -1),
                                     filter_tensor,
                                     strides=strides,
                                     padding=padding)
                        for in_t in input_split]
        result = tf.stack(result_split, axis=3)
        result = tf.squeeze(result, axis=4)
    elif input_tensor.get_shape().as_list()[3] == 1:
        result = tf.nn.conv2d(input_tensor,
                              filter_tensor,
                              strides=strides,
                              padding=padding)
    else:
        raise Exception('nb_channels must be 1 or 3, got %s instead'
                        % str(input_tensor.get_shape().as_list()))

    return result


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):

    """Return the Structural Similarity Map between `img1` and `img2`.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use
            (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel
            (will be reduced for small images).
        k1: Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper).
        k2: Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper).
    Returns:
        Pair containing the mean SSIM and contrast sensitivity between
        `img1` and `img2`.
    Raises:
        RuntimeError: If input images don't have the same shape or don't
        have four dimensions: [batch_size, height, width, depth].
    """

    if img1.get_shape().as_list() != img2.get_shape().as_list():
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).',
            img1.get_shape().as_list(), img2.get_shape().as_list())
    if len(img1.get_shape().as_list()) != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           len(img1.get_shape().as_list()))

    _, height, width, _ = img1.get_shape().as_list()

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = _FSpecialGauss(size, sigma)
        mu1 = _Conv2dIndividual(img1,
                                window,
                                strides=[1, 1, 1, 1],
                                padding='VALID')
        mu2 = _Conv2dIndividual(img2,
                                window,
                                strides=[1, 1, 1, 1],
                                padding='VALID')
        sigma11 = _Conv2dIndividual(img1 * img1,
                                    window,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')
        sigma22 = _Conv2dIndividual(img2 * img2,
                                    window,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')
        sigma12 = _Conv2dIndividual(img1 * img2,
                                    window,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    # return mu1[0, 0, 0, 0], sigma11[0, 0, 0, 0]

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = tf.reduce_mean(((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2))
    cs = tf.reduce_mean(v1 / v2)

    return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):

    """Return the MS-SSIM score between `img1` and `img2`.
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use
            (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel
            (will be reduced for small images).
        k1: Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper).
        k2: Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper).
        weights: List of weights for each level; if none,
            use five levels and the weights from the original paper.
    Returns:
        MS-SSIM score between `img1` and `img2`.
    Raises:
        RuntimeError: If input images don't have the same shape or
            don't have four dimensions: [batch_size, height, width, depth].
    """

    if img1.get_shape().as_list() != img2.get_shape().as_list():
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).',
            img1.get_shape().as_list(), img2.get_shape().as_list())
    if len(img1.get_shape().as_list()) != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           len(img1.get_shape().as_list()))

    # Note: default weights don't sum to 1.0 but do match
    # the paper / matlab code.
    weights = tf.constant(np.array(weights if weights else
                                   [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]),
                          dtype=tf.float32)
    levels = weights.get_shape().as_list()[0]
    downsample_filter = [1, 2, 2, 1]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            img1, img2, max_val=max_val, filter_size=filter_size,
            filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim.append(ssim)
        mcs.append(cs)
        filtered = [tf.nn.avg_pool(im,
                                   downsample_filter,
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
                    for im in [img1, img2]]
        img1, img2 = filtered

    return (tf.reduce_prod(tf.pow(mcs[0:levels - 1], weights[0:levels - 1])) *
            (mssim[levels - 1]**weights[levels - 1]))
