from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from net import network


class mobilenet(network.BaseNetwork):
    def __init__(self, inputs, trainable, conv_basechannel = 1.0, second_channels = None):
        self.base_channels = conv_basechannel
        self.second_channels = second_channels if second_channels else conv_basechannel
        network.BaseNetwork.__init__(inputs=inputs, trainable=trainable)

    def setup(self):
        min_depth = 8
        depth = lambda d: max(int(d * self.base_channels), min_depth)
        depth2 = lambda d: max(int(d * self.second_channels), min_depth)
        with tf.variable_scope(None, 'MobilenetV1'):
            (self.feed('image')
             .convb(3, 3, depth(32), 2, name='Conv2d_0')
             .separable_conv(3, 3, depth(64), 1, name='Conv2d_1')
             .separable_conv(3, 3, depth(128), 2, name='Conv2d_2')
             .separable_conv(3, 3, depth(128), 1, name='Conv2d_3')
             .separable_conv(3, 3, depth(256), 2, name='Conv2d_4')
             .separable_conv(3, 3, depth(256), 1, name='Conv2d_5')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_6')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_7')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_8')
            )
            (self.feed("Conv2d_8")
             .separable_conv(3, 3, depth(1024), 2, name= 'conv2d_dw_3')
             .fc(1501, name='fc_output', relu=False))

    def entropy_softmax_withloss(self, predict, labels):
        output_predict = self.softmax(input=predict, name="predict_feature")
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels*tf.log(output_predict), axis=1))
        return cross_entropy

    def restorable_variable(self):
        vs = {v.op.name: v for v in tf.global_variables() if
              'MobilenetV1/Conv2d' in v.op.name and
              # 'global_step' not in v.op.name and
              # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
              'Ada' not in v.op.name and 'Adam' not in v.op.name
              }
        return vs



