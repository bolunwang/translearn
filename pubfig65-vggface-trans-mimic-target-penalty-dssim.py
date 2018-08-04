#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-03 22:51:25
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import os
import time

import numpy as np
import keras.backend as K
K.set_learning_phase(0)
from keras.models import Model
from keras.models import load_model

import utils_translearn
from attacks_penalty_dssim import MimicPenaltyDSSIM

# for reproducing results
import random
random.seed(12345)
import itertools


##############################
#        PARAMETERS          #
##############################

# parameters about the model

NB_CLASSES = 65
IMG_ROW = 224
IMG_COL = 224
IMG_COLOR = 3
INTENSITY_RANGE = 'imagenet'

# parameters about dataset/model/result path

TEACHER_MODEL_FILE = 'vggface.h5'
STUDENT_MODEL_FILE = 'pubfig65-vggface-trans-nb-train-90.h5'
DATA_FILE = 'pubfig65-imagenet-test.h5'
RESULT_DIR = './pubfig65'

# parameters used for attack

DEVICE = '0'  # which GPU to use

BATCH_SIZE = 1
NB_PAIR = 2

DSSIM_THRESHOLD = 0.003
CUTOFF_LAYER = 38

INITIAL_CONST = 1e10
LR = 1
MAX_ITER = 2000

##############################
#      END PARAMETERS        #
##############################


def load_dataset(data_file=DATA_FILE):

    dataset = utils_translearn.load_dataset(
        data_file,
        keys=['X_test', 'Y_test'])

    X = dataset['X_test']
    Y = dataset['Y_test']

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    X = utils_translearn.preprocess(X, INTENSITY_RANGE)
    Y = np.argmax(Y, axis=1)

    return X, Y


def load_and_build_models(student_model_file=STUDENT_MODEL_FILE,
                          teacher_model_file=TEACHER_MODEL_FILE,
                          cutoff_layer=CUTOFF_LAYER):

    # load the student model
    print('loading student model')
    student_model = load_model(student_model_file)

    print('loading teacher model')
    teacher_model = load_model(teacher_model_file)

    # load the bottleneck model
    print('building bottleneck model')
    bottleneck_model = Model(teacher_model.input,
                             teacher_model.layers[cutoff_layer - 1].output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

    return bottleneck_model, student_model


def filter_data(X, Y, student_model):

    Y_pred = student_model.predict(X)
    accuracy = np.mean(np.argmax(Y_pred, axis=1) == Y)
    print('baseline accuracy of student %f' % accuracy)

    correct_indices = np.argmax(Y_pred, axis=1) == Y
    X = X[correct_indices]
    Y = Y[correct_indices]

    print('X shape: %s' % str(X.shape))
    print('Y shape: %s' % str(Y.shape))

    return X, Y


def select_source_target(X, Y, source, target, sub_sample=(1, 1)):

    # pick source
    source_idx = Y == source
    X_source = X[source_idx]
    Y_source = Y[source_idx]
    # pick targets
    target_idx = Y == target
    X_target = X[target_idx]
    Y_target = Y[target_idx]

    source_random_idx = random.sample(range(X_source.shape[0]), sub_sample[0])
    X_source = X_source[source_random_idx]
    Y_source = Y_source[source_random_idx]

    target_random_idx = random.sample(range(X_target.shape[0]), sub_sample[1])
    X_target = X_target[target_random_idx]
    Y_target = Y_target[target_random_idx]

    return X_source, Y_source, X_target, Y_target


def cal_bottleneck_sim(bottleneck_1, bottleneck_2, weights):

    bottleneck_1 = np.array(bottleneck_1)
    bottleneck_2 = np.array(bottleneck_2)
    bottleneck_1 *= weights
    bottleneck_2 *= weights
    bottleneck_diff = np.abs(bottleneck_1 - bottleneck_2)
    sim = np.sqrt(np.sum(np.square(bottleneck_diff),
                         axis=tuple(range(1, len(bottleneck_1.shape)))))

    return sim


def pubfig65_mimic_penalty_dssim():

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

    sess = utils_translearn.fix_gpu_memory()

    # load models
    (bottleneck_model, student_model) = load_and_build_models()

    # form attacker class
    print('loading attacker')
    attacker = MimicPenaltyDSSIM(sess,
                                 bottleneck_model,
                                 batch_size=BATCH_SIZE,
                                 intensity_range=INTENSITY_RANGE,
                                 initial_const=INITIAL_CONST,
                                 learning_rate=LR,
                                 max_iterations=MAX_ITER,
                                 l_threshold=DSSIM_THRESHOLD,
                                 verbose=1)

    # load dataset for the student model
    print('loading dataset')
    X, Y = load_dataset()

    # filter data points, keep only correctly predicted samples
    print('filtering data')
    X, Y = filter_data(X, Y, student_model)

    # dumping raw images
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    Y_label = list(np.unique(Y))
    all_pair_list = list(itertools.permutations(Y_label, 2))
    pair_list = random.sample(
        all_pair_list,
        min(NB_PAIR, len(all_pair_list)))

    for pair_idx, (source, target) in enumerate(pair_list):

        print('INFO: processing pair #%d source: %d, target: %d'
              % (pair_idx, source, target))

        # sample images
        (X_source, Y_source, X_target, Y_target) = select_source_target(
            X, Y, source, target)

        # LAUNCH ATTACK
        X_adv = attacker.attack(X_source, X_target)

        # test target success
        Y_pred = student_model.predict(X_adv)
        Y_pred = np.argmax(Y_pred, axis=1)
        target_success = np.mean(Y_pred == target)

        # dumping imanges
        img_filename = ('%d-%d.png' % (source, target))
        img_fullpath = '%s/%s' % (RESULT_DIR, img_filename)

        X_source_raw = utils_translearn.reverse_preprocess(
            X_source, INTENSITY_RANGE)
        X_target_raw = utils_translearn.reverse_preprocess(
            X_target, INTENSITY_RANGE)
        X_adv_raw = utils_translearn.reverse_preprocess(
            X_adv, INTENSITY_RANGE)

        grid_img = utils_translearn.generate_grid_img(
            [[X_source_raw[0], X_adv_raw[0], X_target_raw[0]]],
            gap=0, background=0)
        utils_translearn.dump_image(grid_img, img_fullpath, 'png')

        # print final information
        print('INFO: source: %d, target: %d, target_success: %d, img: %s'
              % (source, target, target_success, img_filename))

    pass


if __name__ == '__main__':

    start_time = time.time()
    pubfig65_mimic_penalty_dssim()
    elapsed_time = time.time() - start_time
    print('elapsed time %f s' % (elapsed_time))
