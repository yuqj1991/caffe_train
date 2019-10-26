
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import re


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

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


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def save_tensorflow_ckptmodel(sess, ckpt_dir, global_step=5000): #save ckpt file model, this is old version model saver method
    path = os.path.dirname(os.path.abspath(ckpt_dir))
    if not os.path.isdir(path):
        os.makedirs(path)
    tf.train.Saver().save(sess, ckpt_dir, global_step= global_step) #模型文件后加上"-5000"，model.ckpt-5000.index


def freeze_graph_def(sess, input_graph_def, output_node_names, output_file):
    # input_graph_def = tf.get_default_graph()
    # 1、freeze_graph_def，最重要的就是要确定“指定输出的节点名称”，这个节点名称必须是原模型中存在的节点，对于freeze操作，我们需要定义输出结点的名字。
    # 因为网络其实是比较复杂的，定义了输出结点的名字，那么freeze的时候就只把输出该结点所需要的子图都固化下来，其他无关的就舍弃掉。
    # 因为我们freeze模型的目的是接下来做预测。
    # 所以，output_node_names一般是网络模型最后一层输出的节点名称，或者说就是我们预测的目标。
    # 2、在保存的时候，通过convert_variables_to_constants函数来指定需要固化的节点名称，对于鄙人的代码，需要固化的节点只有一个：output_node_names。
    # 注意节点名称与张量的名称的区别，例如：“input: 0”是张量的名称，而"input"表示的是节点的名称。

    # 3、源码中通过graph = tf.get_default_graph()
    # 获得默认的图，这个图就是由saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    # 恢复的图，因此必须先执行tf.train.import_meta_graph，再执行tf.get_default_graph() 。
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or
                node.name.startswith('image_batch') or node.name.startswith('label_batch') or
                node.name.startswith('phase_train') or node.name.startswith('Logits')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    with tf.gfile.GFile(output_file, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_file))

def center_loss():


def cosin_loss():


def triplet_loss(anchor, postive, negitve, alpha):
    with tf.variable_scope(name_or_scope="triple_loss"):
        pos_distance = tf.reduce_sum(tf.square(anchor - postive), axis=1)
        neg_distance = tf.reduce_sum(tf.square(anchor - negitve), axis=1)
    basic_loss = tf.add(tf.subtract(pos_distance, neg_distance), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0), 0)
    return loss
