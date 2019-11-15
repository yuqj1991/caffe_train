from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import math
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import util_data as utils
from net.config import RNNConfig, CNNConfig
import importlib
import time
import logging
import datetime
import h5py

logging.getLogger().setLevel(logging.INFO)
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# each epoch of one batch_size(), one batch train
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
          control_placeholder, step, loss, cnn_loss, train_op, summary_op, summary_writer,
          stat, accuracy, learning_rate, learning_rate_schedule_file,prelogits, random_rotate, random_crop,
          random_flip, prelogits_hist_max,  use_fixed_image_standardization):
    batch_number = 0
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = utils.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    if lr <= 0:
        return False
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    control_value = utils.RANDOM_ROTATE * random_rotate + utils.RANDOM_CROP * random_crop + utils.RANDOM_FLIP * random_flip \
                    + utils.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                          control_placeholder: control_array})
    # Training loop
    train_time = 0
    start_time = time.time()
    feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                 batch_size_placeholder: args.batch_size}
    tensor_list = [loss, cnn_loss, train_op, step, prelogits, learning_rate, accuracy]
    if epoch % 100 == 0:
        loss_, cnn_loss_, _, step_, prelogits_, lr_, accuracy_, summary_str = sess.run(
            tensor_list + [summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=step_)
    else:
        loss_, cnn_loss_ , _, step_, prelogits_, lr_, accuracy_ = sess.run(
            tensor_list, feed_dict=feed_dict)

    duration = time.time() - start_time
    stat['loss'][step_ - 1] = loss_
    stat['cnn_loss'][step_-1] = cnn_loss_
    stat['learning_rate'][epoch - 1] = lr_
    stat['accuracy'][step_ - 1] = accuracy_
    stat['prelogits_hist'][epoch - 1, :] += \
        np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000,
                     range=(0.0, prelogits_hist_max))[0]

    duration = time.time() - start_time
    print(
        'Epoch: [%d/%d]\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f\tLr %2.5f' %
        (epoch, args.nrof_max_epoch_iters, duration, loss_, accuracy_, lr_))
    train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True


def main(args):
    trainlabelfile = '../../dataset/actiondata/ucf101/ucf_label_101/ucf_101_label_train'
    testlabelfile = '../../dataset/actiondata/ucf101/ucf_label_101/ucf_101_label_test'
    cnnConfig = CNNConfig()
    image_size = (cnnConfig.resized_width, cnnConfig.resized_height)
    rnnConfig = RNNConfig()
    relu = True
    l2_reg_lambea = 0.0
    inception_cnn_output = 101
    # init model
    cnn_net = importlib.import_module(cnnConfig.basemodel)
    l2_loss = tf.constant(0.0)
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    stat_file_name = os.path.join(log_dir, 'stat.h5')
    image_list, label_list = utils.get_image_paths_and_labels(trainlabelfile)
    label_list = list(map(int, label_list))
    utils.shuffle_samples(image_list, label_list)
    assert len(image_list) > 0, 'the image set should not be empty!'
    assert len(image_list) == len(label_list), 'the num of image should equal to the num of label'
    val_image_list, val_label_list = utils.get_image_paths_and_labels(testlabelfile)
    val_label_list = list(map(int, val_label_list))
    utils.shuffle_samples(val_image_list, val_label_list)
    assert len(val_image_list) > 0, 'the image set should not be empty!'
    assert len(val_image_list) == len(val_label_list), 'the num of image should equal to the num of label'
    with tf.Graph().as_default() as graph:
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        if 0:
            test_session = tf.Session()
            test_session.run(tf.global_variables_initializer())
            test_session.run(tf.print(labels))
            print(test_session.run(labels))
            return
        range_size = array_ops.shape(labels)[0]
        # range_size = tf.cast(range_size, dtype=tf.int64)
        # index_queue = tf.data.Dataset.range(range_size).shuffle(range_size)
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)
        index_dequeue_op = index_queue.dequeue_many(cnnConfig.batch_size, 'index_dequeue')
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
        nrof_preprocess_threads = 4
        input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                              dtypes=[tf.string, tf.int32, tf.int32],
                                              shapes=[(1,), (1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder],
                                              name='enqueue_op')
        image_batch, label_batch = utils.create_input_pipeline(input_queue, image_size, nrof_preprocess_threads,
                                                               batch_size_placeholder)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   cnnConfig.learning_rate_decay_epochs * cnnConfig.decay_size,
                                                   cnnConfig.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        with tf.name_scope("cnn_inference"):  # inception_v2_resnet output encode
            prelogits, _ = cnn_net.inference(image_batch, cnnConfig.keep_probability,
                                             phase_train_placeholder,
                                             cnnConfig.embedding_size, cnnConfig.weight_decay)
            logits = slim.fully_connected(prelogits, rnnConfig.num_classes, activation_fn=None,
                                          weights_initializer=slim.initializers.xavier_initializer(),
                                          weights_regularizer=slim.l2_regularizer(cnnConfig.weight_decay),
                                          scope='Logits', reuse=False)
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            # cnn_loss
            prelogits_cnn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits)
            cnn_loss_mean = tf.reduce_mean(prelogits_cnn_loss, name='prelogits_cnn_loss_mean')
        with tf.name_scope("rnn"):  # rnn decode
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnnConfig.hidden_dim)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=rnnConfig.dropout_keep_prob)
            inputs = tf.reshape(embeddings,
                                shape=[batch_size_placeholder, 4, rnnConfig.hidden_dim])
            outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=inputs, dtype=tf.float32)

        with tf.name_scope("result") as scope:  # prediction action 101 action
            outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
            weights = tf.Variable(tf.random_normal([rnnConfig.hidden_dim, rnnConfig.num_classes]))
            biase = tf.Variable(tf.constant(0.1, shape=[rnnConfig.num_classes]))
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            results = op(outputs[-1], weights, biase,
                         name='prediction')  # lstm final output shape.[batch_size, hidden_dim], hidden_dim>101
            correct_prediction = tf.cast(tf.equal(tf.argmax(results, 1), tf.cast(label_batch, tf.int64)), tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
        with tf.name_scope("loss"):
            rnn_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=results)
            rnn_losses_mean = tf.reduce_mean(rnn_losses,name='cross_entropy')
            total_loss = rnn_losses_mean+cnn_loss_mean
            if 0:
                print('image_bach: ', image_batch.shape)
                print('label_batch: ', label_batch.shape)
                print('prelogits. shape:', prelogits.shape)
                print('logits shape: ', logits.shape)
                print('prelogits_cnn_loss: ', prelogits_cnn_loss.shape)
                print('cnn_loss_mean shape: ', cnn_loss_mean.shape)
                print('embeddings, before shape: ', embeddings.shape)
                print('inputs, shape:', inputs.shape)
                print("outputs, before shape: ", outputs.shape)
                print('label_batch shape: ', label_batch.shape)
                print('result shape: ', results.shape)
                print('rnn losses shape: ', rnn_losses.shape)
                print('rnn_losses_mean shape: ', )
                print('total loss shape: ', total_loss.shape)
            # Build a Graph that trains the model with one batch of examples and updates the model parameters
            train_op = utils.train(total_loss, global_step, cnnConfig.optimizer,
                                       learning_rate, cnnConfig.moving_average_decay, tf.global_variables(),
                                       args.log_histograms)

        # create a tensorflow saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # create session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cnnConfig.gpu_memory_fraction)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False,
                                                allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        with sess.as_default():
            # Training and validation loop
            print('Running training')
            nrof_steps = cnnConfig.nrof_max_epoch_iters
            nrof_val_samples = int(math.ceil(
                cnnConfig.nrof_max_epoch_iters / cnnConfig.validate_every_n_epochs))
            # Validate every validate_every_n_epochs as
            # well as in the last epoch
            stat = {
                'loss': np.zeros((nrof_steps,), np.float32),
                'cnn_loss': np.zeros((nrof_steps,), np.float32),
                'val_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
                'accuracy': np.zeros((nrof_steps,), np.float32),
                'learning_rate': np.zeros((cnnConfig.nrof_max_epoch_iters,), np.float32),
                'time_train': np.zeros((cnnConfig.nrof_max_epoch_iters,), np.float32),
                'time_validate': np.zeros((cnnConfig.nrof_max_epoch_iters,), np.float32),
                'prelogits_hist': np.zeros((cnnConfig.nrof_max_epoch_iters, 1000), np.float32),
            }
            for epoch in range(1, cnnConfig.nrof_max_epoch_iters + 1):
                step = sess.run(global_step, feed_dict=None)
                # Train for one epoch
                t = time.time()
                cont = train(cnnConfig, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op,
                             image_paths_placeholder, labels_placeholder,
                             learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                             control_placeholder, global_step, total_loss, cnn_loss_mean, train_op,  summary_op,
                             summary_writer, stat, accuracy, learning_rate, cnnConfig.learning_rate_schedule_file,
                             prelogits, cnnConfig.random_rotate, cnnConfig.random_crop, cnnConfig.random_flip,
                             args.prelogits_hist_max, cnnConfig.use_fixed_image_standardization)
                stat['time_train'][epoch - 1] = time.time() - t

                if not cont:
                    break

                t = time.time()
                if len(val_image_list) > 0 and ((epoch - 1) % cnnConfig.validate_every_n_epochs ==
                                                cnnConfig.validate_every_n_epochs - 1
                                                or epoch == cnnConfig.nrof_max_epoch_iters):
                    validate(cnnConfig, sess, epoch, val_image_list, val_label_list, enqueue_op,
                             image_paths_placeholder,
                             labels_placeholder, control_placeholder,
                             phase_train_placeholder, batch_size_placeholder,
                             stat, total_loss, accuracy,
                             cnnConfig.validate_every_n_epochs, cnnConfig.use_fixed_image_standardization)
                stat['time_validate'][epoch - 1] = time.time() - t

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)

                print('Saving statistics')
                with h5py.File(stat_file_name, 'w') as f:
                    for key, value in stat.items():
                        f.create_dataset(key, data=value)
    return model_dir


def validate(args, sess, epoch, image_list, label_list, enqueue_op, image_paths_placeholder, labels_placeholder,
             control_placeholder, phase_train_placeholder, batch_size_placeholder,
             stat, loss, accuracy, validate_every_n_epochs, use_fixed_image_standardization):
    print('Running forward pass on validation set')
    nrof_batches = len(label_list) // args.val_batch_size
    nrof_images = nrof_batches * args.val_batch_size

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_list[:nrof_images]), 1)
    image_paths_array = np.expand_dims(np.array(image_list[:nrof_images]), 1)
    control_array = np.ones_like(labels_array,
                                 np.int32) * utils.FIXED_STANDARDIZATION * use_fixed_image_standardization
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                          control_placeholder: control_array})

    loss_array = np.zeros((nrof_batches,), np.float32)
    xent_array = np.zeros((nrof_batches,), np.float32)
    accuracy_array = np.zeros((nrof_batches,), np.float32)

    # Training loop
    start_time = time.time()
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: args.val_batch_size}
        loss_, accuracy_ = sess.run([loss, accuracy], feed_dict=feed_dict)
        loss_array[i], accuracy_array[i] = (loss_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time

    val_index = (epoch - 1) // validate_every_n_epochs
    stat['val_loss'][val_index] = np.mean(loss_array)
    stat['val_accuracy'][val_index] = np.mean(accuracy_array)
    print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f' %
          (epoch, duration, np.mean(loss_array), np.mean(accuracy_array)))


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='./logs/action')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='./models/action')
    parser.add_argument('--prelogits_hist_max', type=float,
                        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
