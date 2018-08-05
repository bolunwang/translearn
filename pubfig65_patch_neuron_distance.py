#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-04 16:35:19
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import os
import time
from decimal import Decimal

import numpy as np
import keras.backend as K
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.engine.topology import Layer
from keras.layers import Input
from keras.utils import to_categorical

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
DATA_FILE = 'pubfig65-imagenet.h5'
ADV_DATA_FILE = 'pubfig65-mimic-target-samples-penalty-dssim-0.003.h5'
RESULT_DIR = './pubfig65'

# parameters used for attack

DEVICE = '0'  # which GPU to use

BATCH_SIZE = 32

CUTOFF_LAYER = 38

LR = 0.001  # learning rate of the optimizer
NB_EPOCHS = 50  # total number of epochs of the training
NB_EPOCHS_SUB = 1  # sub steps of epochs for increasing the neuron distance
LOSS_COEFF = 1e-6  # coefficient to balance

CAP = 5 * 1000  # neuron distance threshold
CAP_STEP = 200  # sub steps for increasing neuron distance threshold

##############################
#      END PARAMETERS        #
##############################


class DenseWeightDiff(Layer):

    '''
    This layer calculates the difference between two input vectors.
    This is calculated as part of the objective to measure the distance
    between neuron vectors.
    Thie version weighs neurons using the sum of weights to the next layer
    that connect to each neuron.
    '''

    def __init__(self, other_layer, **kwargs):

        self.other_layer = other_layer
        bottleneck_units = K.int_shape(self.other_layer.kernel)[0]
        self.scale_kernel = K.variable(
            [float(bottleneck_units)] * bottleneck_units,
            dtype=K.floatx())
        super(DenseWeightDiff, self).__init__(**kwargs)

        return

    def call(self, x):

        x1, x2 = x
        sum_kernel = K.sum(K.abs(self.other_layer.kernel), axis=1)
        normalized_kernel = K.l2_normalize(sum_kernel)
        normalized_kernel = normalized_kernel * self.scale_kernel
        output = K.sqrt(K.sum(
            K.square(x1 - x2) * normalized_kernel,
            axis=1))

        return output


# two loss functions used in the patching

def crossentropy(y_true, y_pred):

    from keras.losses import categorical_crossentropy
    entropy = categorical_crossentropy(y_true, y_pred)

    return entropy


def cap_max(y_true, y_pred):

    return K.square(K.maximum(y_true - y_pred, 0))


def recompile_student_model(student_model):

    '''
    Recompile the student model to incorporate the new loss function.
    '''

    # initialize neuron distance layer with weights in the next dense
    dense_weight_diff = DenseWeightDiff(student_model.layers[CUTOFF_LAYER])

    # second input is the original bottleneck neuron vector
    # this value is kept as a fixed input, and used to calculate distance
    bottleneck_shape = student_model.layers[CUTOFF_LAYER - 1].output_shape
    bottleneck_input = Input(bottleneck_shape[1:], name='input_2')

    # initialize the neuron distance layer
    # first input is the fixed neuron vector as the reference point
    # second input is the patched student output at the bottleneck layer
    weighted_diff = dense_weight_diff([
        bottleneck_input,
        student_model.layers[CUTOFF_LAYER - 1].output])

    # compile a new model with multiple inputs and multiple outputs
    model = Model(
        inputs=[student_model.input,
                bottleneck_input],
        outputs=[student_model.output,
                 weighted_diff])

    # compile with optimizer and a coefficient to balance two loss terms
    # first loss is a simple cross entropy
    # second loss is a capped max loss, which only becomes positive when
    # the predict value (actual neuron distance) is smaller than the
    # true value desired (pre-determined neuron distance threshold).
    # The original formulation is a constrained optimization problem.
    # We use penalty method to convert this into a multi objective
    # optimization problem.
    # COEFF is the penalty cost. Setting it to the appropriate value can
    # make sure the opmization will converge to a point where the
    # second term (constraint) will be met, while the first term can be
    # optimized.
    optimizer = SGD(lr=LR)
    model.compile(loss=[crossentropy, cap_max],
                  loss_weights=[1, LOSS_COEFF],
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def load_and_build_models(student_model_file=STUDENT_MODEL_FILE,
                          teacher_model_file=TEACHER_MODEL_FILE,
                          cutoff_layer=CUTOFF_LAYER):

    # load the student model
    print('loading student model')
    student_model = load_model(student_model_file)

    for idx, layer in enumerate(student_model.layers):
        layer.trainable = True

    print('loading teacher model')
    teacher_model = load_model(teacher_model_file)

    # load the bottleneck model
    print('building bottleneck model')
    bottleneck_model = Model(teacher_model.input,
                             teacher_model.layers[cutoff_layer - 1].output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

    student_model = recompile_student_model(student_model)

    return bottleneck_model, student_model


def load_dataset(data_file=DATA_FILE):

    dataset = utils_translearn.load_dataset(
        data_file,
        keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    X_train = utils_translearn.preprocess(X_train, INTENSITY_RANGE)
    X_test = utils_translearn.preprocess(X_test, INTENSITY_RANGE)

    print('X_train shape: %s' % str(X_train.shape))
    print('Y_train shape: %s' % str(Y_train.shape))
    print('X_test shape: %s' % str(X_test.shape))
    print('Y_test shape: %s' % str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test


def load_adv_image(adv_data_file=ADV_DATA_FILE):

    dataset = utils_translearn.load_dataset(
        adv_data_file,
        keys=['X_source', 'X_adv', 'Y_source', 'Y_target'])
    X_source = dataset['X_source']
    X_adv = dataset['X_adv']
    Y_source = dataset['Y_source']
    Y_target = dataset['Y_target']

    Y_source = to_categorical(Y_source, NB_CLASSES)
    Y_target = to_categorical(Y_target, NB_CLASSES)

    print('X_adv shape: %s' % str(X_adv.shape))
    print('Y_source shape: %s' % str(Y_source.shape))

    return X_source, X_adv, Y_source, Y_target


def train_model(model, X_train, Y_train,
                cap=CAP, cap_step=CAP_STEP,
                nb_epochs_sub=NB_EPOCHS_SUB, nb_epochs=NB_EPOCHS):

    '''
    We use an incremental approach to stablize the training process.training
    After each nb_epochs_sub, we increase the neuron distance threshold
    (current_cap) by cap_step, until we reach the final threshold (cap).
    This function prints out the intermediate log of the training result.
    cap shows the current neuron distance threshold used in this epoch.
    loss_total is the total loss, loss_ce is the cross entropy, loss_dis
    is the current distance away from the desired neuron distance threshold
    (temporary threshold). raw shows the raw distance. acc shows the
    classification accuracy on the training dataset.
    '''

    callbacks = []

    current_cap = 0

    for epoch in range(nb_epochs):

        if epoch % nb_epochs_sub == 0:
            if current_cap < cap:
                current_cap += cap_step
                Y_train = reset_cap(Y_train, current_cap)

        history = model.fit(X_train, Y_train,
                            epochs=1,
                            verbose=0,
                            callbacks=callbacks)
        logs = history.history

        loss_total = logs['loss'][0]
        loss_ce = logs['activation_17_loss'][0]
        loss_dis = logs['dense_weight_diff_1_loss'][0],
        loss_dis = np.sqrt(loss_dis)
        raw_dis = current_cap - loss_dis
        dis_over = loss_dis / current_cap * 100
        acc = logs['activation_17_acc'][0]
        print('epoch %04d/%04d; cap: %.2E, loss_total: %f, loss_ce: %f, loss_dis: %.2f (raw: %.2f, %.1f%% less), acc: %.4f' %
              (epoch, NB_EPOCHS, Decimal(current_cap),
               loss_total, loss_ce, loss_dis, raw_dis, dis_over, acc))

    return model


def transform_dataset(bottleneck_model, X, Y, cap):

    '''
    Transform the original training data to add bottleneck neuron vector
    into X, and adding the neuron distance threshold (cap) to Y.
    '''

    Y_pred = bottleneck_model.predict(X)

    X_bottleneck = Y_pred

    X = [X, X_bottleneck]
    Y = [Y, np.array([cap] * Y.shape[0])]

    return X, Y


def reset_cap(Y, cap):

    # resetting neuron distance threshold
    Y[1] = np.array([cap] * Y[0].shape[0])

    return Y


def eval_model(student_model, X_test, Y_test, X_adv, Y_adv):

    # evaluate model classification accuracy and attack success rate
    Y_pred = student_model.predict(X_test)
    correct_indices = (
        np.argmax(Y_pred[0], axis=1) ==
        np.argmax(Y_test[0], axis=1))
    test_acc = np.mean(correct_indices)
    test_neuron_sim = np.mean(Y_pred[1])

    Y_pred = student_model.predict(X_adv)
    correct_indices = (
        np.argmax(Y_pred[0], axis=1) ==
        np.argmax(Y_adv[0], axis=1))
    attack_success = np.mean(correct_indices)
    print('INFO: acc: %f, attack: %f, sim: %f' %
          (test_acc, attack_success, test_neuron_sim))

    return


def pubfig65_patch_neuron_distance():

    # specify which GPU to use for training
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils_translearn.fix_gpu_memory()

    print('loading models')
    bottleneck_model, student_model = load_and_build_models()

    print('loading dataset')
    X_train, Y_train, X_test, Y_test = load_dataset()
    X_source, X_adv, Y_source, Y_target = load_adv_image()

    # modify data to include bottleneck neuron values in Teacher (into X) and
    # add neuron distance threshold (into Y)
    print('transforming datasets')
    X_train, Y_train = transform_dataset(bottleneck_model, X_train, Y_train, 0)
    X_test, Y_test = transform_dataset(bottleneck_model, X_test, Y_test, 0)
    X_adv, Y_target = transform_dataset(bottleneck_model, X_adv, Y_target, 0)

    # evaluate model performance before training
    eval_model(student_model, X_test, Y_test, X_adv, Y_target)

    # model training
    student_model = train_model(student_model, X_train, Y_train,
                                cap=CAP, cap_step=CAP_STEP)

    # evaluate model performance after training
    eval_model(student_model, X_test, Y_test, X_adv, Y_target)

    return


if __name__ == '__main__':

    start_time = time.time()
    pubfig65_patch_neuron_distance()
    elapsed_time = time.time() - start_time
    print('elapsed time %f s' % (elapsed_time))
