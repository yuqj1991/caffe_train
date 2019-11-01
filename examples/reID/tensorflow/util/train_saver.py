# -*- coding: utf-8 -*-
import re
import tensorflow as tf
import numpy as np
import os

# save model
# restore model
# freeze graph to pb file
# 当你已经训练好一个神经网络之后，你想要保存它，用于以后的使用，部署到产品里面去。所以，Tensorflow模型是什么？
# Tensorflow模型主要包含网络的设计或者图（graph），和我们已经训练好的网络参数的值。因此Tensorflow模型有两个主要的文件：
# A） Meta graph:
# 这是一个保存完整Tensorflow graph的protocol buffer，比如说，所有的 variables, operations, collections等等。这个文件的后缀是 .meta 。
# B） Checkpoint file:
# 这是一个包含所有权重（weights），偏置（biases），梯度（gradients）和所有其他保存的变量（variables）的二进制文件。它包含两个文件：
# mymodel.data-00000-of-00001
# mymodel.index
# 其中，.data文件包含了我们的训练变量。另外，除了这两个文件，Tensorflow有一个叫做checkpoint的文件，记录着已经最新的保存的模型文件。
# 注：Tensorflow 0.11版本以前，Checkpoint file只有一个后缀名为.ckpt的文件。

# 我们要保存所有变量和操作： saver = tf.train.Saver()
# 由于网络的图（graph）在训练的时候是不会改变的，因此，我们没有必要每次都重复保存.meta文件，可以使用如下方法：
# saver.save(sess, 'my-model',global_step=step,write_meta_graph=False)
# 注意到，我们在tf.train.Saver()中并没有指定任何东西，因此它将保存所有变量。如果我们不想保存所有的变量，只想保存其中一些变量，
# 我们可以在创建tf.train.Saver实例的时候，给它传递一个我们想要保存的变量的list或者字典。 var_list=[]
# A）创建网络 saver = tf.train.import_meta_graph('my_test_model-1000.meta') ,
# 注意，上面仅仅是将已经定义的网络导入到当前的graph中，但是我们还是需要加载网络的参数值。
# B）加载参数Load the parameters， 我们可以通过调用restore函数来恢复网络的参数

# ps，重要，如何恢复任何一个预训练好的模型，并使用它来预测，fine-tuning或者进一步训练。当你使用Tensorflow时，你会定义一个图（graph），
# 其中，你会给这个图喂（feed）训练数据和一些超参数（比如说learning rate，global step等）。比如我们使用的是已经预训练好的模型，来调优新的数据集
# 我们通过saver = tf.train.import_meta_graph('my_test_model-1000.meta')和restore恢复了网络图，和参数，权重，偏置，操作op节点，
# 在这之后，我们可以继续添加新的操作节点，产生新的变量，只训练新的变量，获取到相应的操作节点，tensor，这样可以知道tensor，可以引入新的变量对其进行训练
# we can just train some new variables,just make trainable = True, and make the trained fine weights make them trainable
# = False

# checkpoint 是什么？
# checkpoint是一个文本文件，记录了训练过程中在所有中间节点上保存的模型的名称，首行记录的是最后（最近）一次保存的模型名称。
# checkpoint是检查点文件，文件保存了一个目录下所有的模型文件列表


class SaverTensorflow(object):
    def __init__(self, save_dir, sess, input_map = None):
        self.dir = save_dir
        self.sess = sess
        self.input_map = input_map

    def get_model_filenames(self):
        files = os.listdir(self.dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % self.dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % self.dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(self.dir)
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

    def save_tensorflow_ckptmodel(self,
                                  global_step=5000):  # save ckpt file model, this is old version model saver method
        path = os.path.dirname(os.path.abspath(self.dir))
        if not os.path.isdir(path):
            os.makedirs(path)
        tf.train.Saver().save(self.sess, self.dir, global_step=global_step)  # 模型文件后加上"-5000"，model.ckpt-5000.index

    def load(self, input_map=None):
        model_exp = os.path.expanduser(self.dir)
        if os.path.isfile(model_exp):
            print('Model filename: %s' % model_exp)
            with tf.gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=input_map, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = self.get_model_filenames(model_exp)
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    # 这个需要根据特殊情况，进行特殊指定， 与其图像节点有关
    def freeze_graph_def(self,sess, input_graph_def, output_node_names, output_file):
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