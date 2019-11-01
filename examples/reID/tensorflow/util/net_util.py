
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import re
from util import train_summary_log


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    summary_log = train_summary_log(log_dir="log",loss=total_loss)
    loss_averages_op = summary_log.add_loss_summaries(collection_name='losses')

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
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

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def center_loss(features, label, alfa, nrof_classes):
    """
    Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def cosin_loss(anchor, postive, negitve, alpha, eps):
    with tf.variable_scope(name_or_scope="cosin_loss"):
        pos_cos_distance = tf.reduce_sum(tf.multiply(anchor, postive)/(tf.sqrt(tf.square(anchor)) * tf.sqrt(tf.square(postive)) + eps), axis=1)
        neg_cos_distance = tf.reduce_sum(tf.multiply(anchor, negitve)/(tf.sqrt(tf.square(anchor)) * tf.sqrt(tf.square(negitve)) + eps), axis=1)
    basic_loss = tf.add(tf.subtract(pos_cos_distance, neg_cos_distance), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0), 0)
    return loss


def triplet_loss(anchor, postive, negitve, alpha):
    with tf.variable_scope(name_or_scope="triple_loss"):
        pos_distance = tf.reduce_sum(tf.square(anchor - postive), axis=1)
        neg_distance = tf.reduce_sum(tf.square(anchor - negitve), axis=1)
    basic_loss = tf.add(tf.subtract(pos_distance, neg_distance), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0), 0)
    return loss
