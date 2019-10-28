import math

import numpy as np
import tensorflow as tf

Batch_size = 64
MAX_EPOCHES = 120000
learning_rate = 0.01
learning_policy = 'MultiStep'
MultiStepValue = [20000, 100000, 300000]
CurrentAdjustStep = 0

# set learning rate policy adjust
# set data parall
# set model net
# add train log loss, evaluation
# set train optimizer gradient
# save model to file


def adjust_learning_rate_by_policy(base_lr_rate, gamma, iter, step, policy):
    if policy == 'MultiStep':
        global CurrentAdjustStep
        if CurrentAdjustStep < len(MultiStepValue) and iter > MultiStepValue[CurrentAdjustStep]:
            CurrentAdjustStep += 1
        return base_lr_rate * math.pow(gamma, CurrentAdjustStep)
    elif policy == 'fixed':
        return base_lr_rate
    elif policy == 'step':
        return base_lr_rate *math.pow(gamma, math.floor(iter/step))
    elif policy == 'exp':
        return base_lr_rate*math.pow(gamma, iter)


def adjust_optimizer_by_tensorflow(optimizer, learning_rate):
    if optimizer == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    return opt


def train(sess, dataset, loss, optimizer_op, ):


def evaluate(sess, ):




