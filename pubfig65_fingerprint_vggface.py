#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-07 10:16:40
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import os
import time

import numpy as np
import keras.backend as K
K.set_learning_phase(0)
from keras.models import Model

from mimic_penalty_dssim import MimicPenaltyDSSIM

import random
random.seed(1234)

from keras.models import load_model

import utils_translearn


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

# parameters used for attack

DEVICE = '0'  # which GPU to use

BATCH_SIZE = 1
NB_IMGS = 1

DSSIM_THRESHOLD = 1
CUTOFF_LAYER = 38

INITIAL_CONST = 1e10
LR = 1
MAX_ITER = 2000

##############################
#      END PARAMETERS        #
##############################


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


def pubfig65_fingerprint_vggface():

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

    sess = utils_translearn.fix_gpu_memory()

    bottleneck_model, student_model = load_and_build_models()

    # form attacker class
    # setting mimic_img to False, since we are mimicking a specific vector
    print('loading attacker')
    attacker = MimicPenaltyDSSIM(sess,
                                 bottleneck_model,
                                 mimic_img=False,
                                 batch_size=BATCH_SIZE,
                                 intensity_range=INTENSITY_RANGE,
                                 initial_const=INITIAL_CONST,
                                 learning_rate=LR,
                                 max_iterations=MAX_ITER,
                                 l_threshold=DSSIM_THRESHOLD,
                                 verbose=1)

    print('building fingerprinting input')
    # initializing input with random noise
    X_source_raw = np.random.random((NB_IMGS, IMG_ROW, IMG_COL, IMG_COLOR))
    X_source_raw *= 255.0
    X_source = utils_translearn.preprocess(X_source_raw, INTENSITY_RANGE)

    # build target bottleneck neuron vector as all-zero vector
    bottleneck_shape = (
        [X_source.shape[0]] +
        list(bottleneck_model.layers[-1].output_shape[1:]))
    X_target_bottleneck = np.zeros(bottleneck_shape)

    # build fingerprinting input
    X_adv = attacker.attack(X_source, X_target_bottleneck)

    print('testing fingerprint image on student')
    gini_list = []
    max_conf_list = []
    Y_pred = student_model.predict(X_adv)
    Y_conf = np.max(Y_pred, axis=1)
    for idx in xrange(NB_IMGS):
        gini_list.append(utils_translearn.gini(Y_pred[idx]))
        max_conf_list.append(Y_conf[idx])

    avg_gini = np.mean(gini_list)
    avg_max_conf = np.mean(max_conf_list)
    print('INFO: avg_gini: %f, avg_max_conf: %f' % (avg_gini, avg_max_conf))

    pass


if __name__ == '__main__':

    start_time = time.time()
    pubfig65_fingerprint_vggface()
    elapsed_time = time.time() - start_time
    print('elapsed time %f s' % (elapsed_time))
