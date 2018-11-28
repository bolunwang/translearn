#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-03 22:50:49
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import numpy as np
import tensorflow as tf
from six.moves import xrange

import datetime
import time
from decimal import Decimal

from utils_translearn import preprocess, reverse_preprocess, cal_rmsd
from msssim_tf import MultiScaleSSIM


class MimicPenaltyDSSIM:

    # if the attack is trying to mimic a target image or a neuron vector
    MIMIC_IMG = True
    # number of iterations to perform gradient descent
    MAX_ITERATIONS = 10000
    # larger values converge faster to less accurate results
    LEARNING_RATE = 1e-2
    # the initial constant c to pick as a first guess
    INITIAL_CONST = 1
    # pixel intensity range
    INTENSITY_RANGE = 'imagenet'
    # threshold for distance
    L_THRESHOLD = 0.03
    # whether keep the final result or the best result
    KEEP_FINAL = False
    # max_val of image
    MAX_VAL = 255
    # The following variables are used by DSSIM, should keep as default
    # filter size in SSIM
    FILTER_SIZE = 11
    # filter sigma in SSIM
    FILTER_SIGMA = 1.5
    # weights used in MS-SSIM
    SCALE_WEIGHTS = None

    def __init__(self, sess, bottleneck_model, mimic_img=MIMIC_IMG,
                 batch_size=1, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, initial_const=INITIAL_CONST,
                 intensity_range=INTENSITY_RANGE, l_threshold=L_THRESHOLD,
                 max_val=MAX_VAL, keep_final=KEEP_FINAL,
                 filter_size=FILTER_SIZE, filter_sigma=FILTER_SIGMA,
                 scale_weights=SCALE_WEIGHTS,
                 verbose=0):
        """
        The L_2 optimized attack.
        Returns adversarial examples for the supplied bottleneck model.
        sess: A TensorFlow session
        bottleneck_model: The teacher model cut off at the layer attacker
            tries to launch the mimicry attack
        mimic_img: Whether the attack is trying to mimic a target image
            or a bottleneck neuron vector directly.
        batch_size: Number of attacks to run simultaneously.
        learning_rate: The learning rate for the attack algorithm.
            Smaller values produce better results but are slower to converge.
        max_iterations: The maximum number of iterations.
            Larger values are more accurate; setting too small will require
            a large learning rate and will produce poor results.
        initial_const: The initial tradeoff-constant to use to tune the
            relative importance of DSSIM and bottleneck similarity.
        intensity_range: The preprocessing method used by the model.
            'raw': no preprocessing
            'imagenet': default imagenet mean centering
            'inception': inception preprocessing, scaling to [-1, 1]
            'mnist': scaling to [0, 1]
        l_threshold: The DSSIM budget used for crafting adversarial samples.
        max_val: Maximum pixel intensity. Default set to 255.
        keep_final: Whether to keep the final optimization result or keep the
            best result throughout the process.
        verbose: Whether to output the intermediate log or not.
        """

        assert intensity_range in {'raw', 'imagenet', 'inception', 'mnist'}

        # constant used for tanh transformation to avoid corner cases
        self.tanh_constant = 2 - 1e-6
        self.sess = sess
        self.MIMIC_IMG = mimic_img
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.intensity_range = intensity_range
        self.l_threshold = l_threshold
        self.max_val = max_val
        self.keep_final = keep_final
        self.verbose = verbose

        self.input_shape = tuple(
            [self.batch_size] +
            list(bottleneck_model.input_shape[1:]))

        self.bottleneck_shape = tuple(
            [self.batch_size] +
            list(bottleneck_model.output_shape[1:]))

        '''
        VARIABLE ASSIGNMENT
        '''

        # the variable we're going to optimize over
        self.modifier = tf.Variable(
            np.zeros(self.input_shape, dtype=np.float32))

        # target image in tanh space
        if self.MIMIC_IMG:
            self.timg_tanh = tf.Variable(
                np.zeros(self.input_shape), dtype=np.float32)
        else:
            self.bottleneck_t_raw = tf.Variable(
                np.zeros(self.bottleneck_shape), dtype=np.float32)
        # source image in tanh space
        self.simg_tanh = tf.Variable(
            np.zeros(self.input_shape), dtype=np.float32)
        self.const = tf.Variable(np.ones(batch_size), dtype=np.float32)
        self.mask = tf.Variable(np.ones((batch_size), dtype=np.bool))
        self.weights = tf.Variable(np.ones(self.bottleneck_shape,
                                           dtype=np.float32))

        # and here's what we use to assign them
        self.assign_modifier = tf.placeholder(tf.float32, self.input_shape)
        if self.MIMIC_IMG:
            self.assign_timg_tanh = tf.placeholder(
                tf.float32, self.input_shape)
        else:
            self.assign_bottleneck_t_raw = tf.placeholder(
                tf.float32, self.bottleneck_shape)
        self.assign_simg_tanh = tf.placeholder(tf.float32, self.input_shape)
        self.assign_const = tf.placeholder(tf.float32, (batch_size))
        self.assign_mask = tf.placeholder(tf.bool, (batch_size))
        self.assign_weights = tf.placeholder(tf.float32, self.bottleneck_shape)

        '''
        PREPROCESSING
        '''

        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        # adversarial image in raw space
        self.aimg_raw = (tf.tanh(self.modifier + self.simg_tanh) /
                         self.tanh_constant +
                         0.5) * 255.0
        # source image in raw space
        self.simg_raw = (tf.tanh(self.simg_tanh) /
                         self.tanh_constant +
                         0.5) * 255.0
        if self.MIMIC_IMG:
            # target image in raw space
            self.timg_raw = (tf.tanh(self.timg_tanh) /
                             self.tanh_constant +
                             0.5) * 255.0

        # convert source and adversarial image into input space
        if self.intensity_range == 'imagenet':
            mean = tf.constant(np.repeat([[[[103.939, 116.779, 123.68]]]],
                                         self.batch_size,
                                         axis=0),
                               dtype=tf.float32,
                               name='img_mean')
            self.aimg_input = (self.aimg_raw[..., ::-1] - mean)
            self.simg_input = (self.simg_raw[..., ::-1] - mean)
            if self.MIMIC_IMG:
                self.timg_input = (self.timg_raw[..., ::-1] - mean)

        elif self.intensity_range == 'inception':
            self.aimg_input = (self.aimg_raw / 255.0 - 0.5) * 2.0
            self.simg_input = (self.simg_raw / 255.0 - 0.5) * 2.0
            if self.MIMIC_IMG:
                self.timg_input = (self.timg_raw / 255.0 - 0.5) * 2.0

        elif self.intensity_range == 'mnist':
            self.aimg_input = self.aimg_raw / 255.0
            self.simg_input = self.simg_raw / 255.0
            if self.MIMIC_IMG:
                self.timg_input = self.timg_raw / 255.0

        elif self.intensity_range == 'raw':
            self.aimg_input = self.aimg_raw
            self.simg_input = self.simg_raw
            if self.MIMIC_IMG:
                self.timg_input = self.timg_raw

        '''
        CONSTRAINTS: perturbation
        DSSIM: structural dis-similarity between two images.
        '''

        def batch_gen_DSSIM(aimg_raw_split, simg_raw_split):

            msssim_split = [
                MultiScaleSSIM(
                    tf.expand_dims(aimg_raw_split[idx], 0),
                    tf.expand_dims(simg_raw_split[idx], 0),
                    max_val=max_val, filter_size=filter_size,
                    filter_sigma=filter_sigma, weights=scale_weights)
                for idx in xrange(batch_size)]

            dssim = (1.0 - tf.stack(msssim_split)) / 2.0

            return dssim

        # unstack tensor into list to calculate DSSIM
        aimg_raw_split = tf.unstack(self.aimg_raw, batch_size, axis=0)
        simg_raw_split = tf.unstack(self.simg_raw, batch_size, axis=0)

        # raw value of DSSIM distance
        self.dist_raw = batch_gen_DSSIM(aimg_raw_split, simg_raw_split)
        # distance value after applying threshold
        self.dist = tf.maximum(self.dist_raw - self.l_threshold, 0.0)

        self.dist_raw_sum = tf.reduce_sum(
            tf.where(self.mask,
                     self.dist_raw,
                     tf.zeros_like(self.dist_raw)))
        self.dist_sum = tf.reduce_sum(tf.where(self.mask,
                                               self.dist,
                                               tf.zeros_like(self.dist)))

        '''
        BOTTLESIM
        similarity between neuron values between new images and original images
        '''

        self.bottleneck_a = bottleneck_model(self.aimg_input)
        self.bottleneck_a *= self.weights
        if self.MIMIC_IMG:
            self.bottleneck_t = bottleneck_model(self.timg_input)
            self.bottleneck_t *= self.weights
        else:
            self.bottleneck_t = self.weights * self.bottleneck_t_raw

        # L2 diff between two sets of non-normalized bottleneck neurons
        # L2 diff = sqrt(sum(square(X - Y)))
        # Careful about the dimensions when applying math operators
        bottleneck_diff = self.bottleneck_t - self.bottleneck_a
        self.bottlesim = tf.sqrt(
            tf.reduce_sum(tf.square(bottleneck_diff),
                          axis=range(1, len(self.bottleneck_shape))))

        self.bottlesim_sum = tf.reduce_sum(
            tf.where(self.mask,
                     self.bottlesim,
                     tf.zeros_like(self.bottlesim)))

        '''
        Sum up two losses
        '''

        # sum up the losses
        self.loss = self.const * tf.square(self.dist) + self.bottlesim

        self.loss_sum = tf.reduce_sum(tf.where(self.mask,
                                               self.loss,
                                               tf.zeros_like(self.loss)))

        '''
        Setup phase
        '''

        # Setup the Adadelta optimizer and keep track of variables
        # we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdadeltaOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss_sum,
                                        var_list=[self.modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.modifier.assign(self.assign_modifier))
        if self.MIMIC_IMG:
            self.setup.append(self.timg_tanh.assign(self.assign_timg_tanh))
        else:
            self.setup.append(self.bottleneck_t_raw.assign(
                self.assign_bottleneck_t_raw))
        self.setup.append(self.simg_tanh.assign(self.assign_simg_tanh))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.mask.assign(self.assign_mask))
        self.setup.append(self.weights.assign(self.assign_weights))

        self.init = tf.variables_initializer(
            var_list=[self.modifier] + new_vars)

        print('Attacker loaded')

    def preprocess_arctanh(self, imgs):

            imgs = reverse_preprocess(imgs, self.intensity_range)
            imgs /= 255.0
            imgs -= 0.5
            imgs *= self.tanh_constant
            tanh_imgs = np.arctanh(imgs)

            return tanh_imgs

    def clipping(self, imgs):

        imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs = np.clip(imgs, 0, self.max_val)
        imgs = np.rint(imgs)

        imgs = preprocess(imgs, self.intensity_range)

        return imgs

    def print_stat(self, source_imgs, best_adv):

        avg_rmsd, std_rmsd = cal_rmsd(source_imgs, best_adv)
        print('Avg RMSD: %.4f, STD RMSD: %.4f' % (avg_rmsd, std_rmsd))

        return

    def attack(self, source_imgs, target_imgs, weights=None):

        # weights defines
        if weights is None:
            weights = np.ones([source_imgs.shape[0]] +
                              list(self.bottleneck_shape[1:]))

        assert weights.shape[1:] == self.bottleneck_shape[1:]
        assert source_imgs.shape[1:] == self.input_shape[1:]
        assert source_imgs.shape[0] == weights.shape[0]
        if self.MIMIC_IMG:
            assert target_imgs.shape[1:] == self.input_shape[1:]
            assert source_imgs.shape[0] == target_imgs.shape[0]
        else:
            # target_imgs should be bottleneck values
            # we do not rename the variable here
            assert target_imgs.shape[1:] == self.bottleneck_shape[1:]
            assert source_imgs.shape[0] == target_imgs.shape[0]

        start_time = time.time()

        adv_imgs = []
        print('%d batches in total'
              % int(np.ceil(len(source_imgs) / self.batch_size)))

        for idx in range(0, len(source_imgs), self.batch_size):
            print('processing batch %d at %s' % (idx, datetime.datetime.now()))
            adv_img = self.attack_batch(source_imgs[idx:idx + self.batch_size],
                                        target_imgs[idx:idx + self.batch_size],
                                        weights[idx:idx + self.batch_size])
            adv_imgs.extend(adv_img)

        elapsed_time = time.time() - start_time
        print('attack cost %f s' % (elapsed_time))

        return np.array(adv_imgs)

    def attack_batch(self, source_imgs, target_imgs, weights):

        """
        Run the attack on a batch of images and labels.
        """

        # if self.verbose == 1:
        #     print('imgs shape %s' % str(imgs.shape))
        #     print('target_imgs shape %s' % str(target_imgs.shape))
        #     print('weights shape %s' % str(weights.shape))

        nb_imgs = source_imgs.shape[0]
        mask = [True] * nb_imgs + [False] * (self.batch_size - nb_imgs)
        mask = np.array(mask, dtype=np.bool)

        source_imgs = np.array(source_imgs)
        target_imgs = np.array(target_imgs)

        # convert to tanh-space
        simg_tanh = self.preprocess_arctanh(source_imgs)
        if self.MIMIC_IMG:
            timg_tanh = self.preprocess_arctanh(target_imgs)
        else:
            timg_tanh = target_imgs

        CONST = np.ones(self.batch_size) * self.initial_const

        self.sess.run(self.init)
        simg_tanh_batch = np.zeros(self.input_shape)
        if self.MIMIC_IMG:
            timg_tanh_batch = np.zeros(self.input_shape)
        else:
            timg_tanh_batch = np.zeros(self.bottleneck_shape)
        weights_batch = np.zeros(self.bottleneck_shape)
        simg_tanh_batch[:nb_imgs] = simg_tanh[:nb_imgs]
        timg_tanh_batch[:nb_imgs] = timg_tanh[:nb_imgs]
        weights_batch[:nb_imgs] = weights[:nb_imgs]
        modifier_batch = np.ones(self.input_shape) * 1e-6

        # set the variables so that we don't have to send them over again
        if self.MIMIC_IMG:
            self.sess.run(self.setup,
                          {self.assign_timg_tanh: timg_tanh_batch,
                           self.assign_simg_tanh: simg_tanh_batch,
                           self.assign_const: CONST,
                           self.assign_mask: mask,
                           self.assign_weights: weights_batch,
                           self.assign_modifier: modifier_batch})
        else:
            # if directly mimicking a vector, use assign_bottleneck_t_raw
            # in setup
            self.sess.run(self.setup,
                          {self.assign_bottleneck_t_raw: timg_tanh_batch,
                           self.assign_simg_tanh: simg_tanh_batch,
                           self.assign_const: CONST,
                           self.assign_mask: mask,
                           self.assign_weights: weights_batch,
                           self.assign_modifier: modifier_batch})

        # if self.verbose == 1:
        #     print('************************************************')
        #     print('CONST: %s' % str(CONST))

        # the best bottlesim and adv img
        best_bottlesim = [np.inf] * nb_imgs
        best_adv = np.zeros_like(source_imgs)

        if self.verbose == 1:
            loss_sum = float(self.sess.run(self.loss_sum))
            dist_sum = float(self.sess.run(self.dist_sum))
            thresh_over = (dist_sum / self.batch_size / self.l_threshold * 100)
            dist_raw_sum = float(self.sess.run(self.dist_raw_sum))
            bottlesim_sum = self.sess.run(self.bottlesim_sum)
            print('START:     Total loss: %.4E; perturb: %.6f (%.2f%% over, raw: %.6f); sim: %f'
                  % (Decimal(loss_sum),
                     dist_sum,
                     thresh_over,
                     dist_raw_sum,
                     bottlesim_sum))

        for iteration in xrange(self.MAX_ITERATIONS):

            self.sess.run([self.train])

            dist_raw_list, bottlesim_list, aimg_input_list = self.sess.run(
                [self.dist_raw,
                 self.bottlesim,
                 self.aimg_input])
            for e, (dist_raw, bottlesim, aimg_input) in enumerate(
                    zip(dist_raw_list, bottlesim_list, aimg_input_list)):
                if e >= nb_imgs:
                    break
                if (dist_raw < self.l_threshold and
                        bottlesim < best_bottlesim[e]):
                    best_bottlesim[e] = bottlesim
                    best_adv[e] = aimg_input

            # print out the losses every 10%
            if iteration % (self.MAX_ITERATIONS // 10) == 0:
                if self.verbose == 1:
                    loss_sum = float(self.sess.run(self.loss_sum))
                    dist_sum = float(self.sess.run(self.dist_sum))
                    thresh_over = (dist_sum /
                                   self.batch_size /
                                   self.l_threshold *
                                   100)
                    dist_raw_sum = float(self.sess.run(self.dist_raw_sum))
                    bottlesim_sum = self.sess.run(self.bottlesim_sum)
                    print('ITER %4d: Total loss: %.4E; perturb: %.6f (%.2f%% over, raw: %.6f); sim: %f'
                          % (iteration,
                             Decimal(loss_sum),
                             dist_sum,
                             thresh_over,
                             dist_raw_sum,
                             bottlesim_sum))

        if self.verbose == 1:
            loss_sum = float(self.sess.run(self.loss_sum))
            dist_sum = float(self.sess.run(self.dist_sum))
            thresh_over = (dist_sum / self.batch_size / self.l_threshold * 100)
            dist_raw_sum = float(self.sess.run(self.dist_raw_sum))
            bottlesim_sum = float(self.sess.run(self.bottlesim_sum))
            print('END:       Total loss: %.4E; perturb: %.6f (%.2f%% over, raw: %.6f); sim: %f'
                  % (Decimal(loss_sum),
                     dist_sum,
                     thresh_over,
                     dist_raw_sum,
                     bottlesim_sum))

        # keep the final round result
        if self.keep_final:
            dist_raw_list, bottlesim_list, aimg_input_list = self.sess.run(
                [self.dist_raw,
                 self.bottlesim,
                 self.aimg_input])
            for e, (dist_raw, bottlesim, aimg_input) in enumerate(
                    zip(dist_raw_list, bottlesim_list, aimg_input_list)):
                if e >= nb_imgs:
                    break
                if (dist_raw < self.l_threshold and
                        bottlesim < best_bottlesim[e]):
                    best_bottlesim[e] = bottlesim
                    best_adv[e] = aimg_input

        if self.verbose == 1:
            self.print_stat(source_imgs[:nb_imgs], best_adv[:nb_imgs])

        # return the best solution found
        best_adv = self.clipping(best_adv[:nb_imgs])

        return best_adv
