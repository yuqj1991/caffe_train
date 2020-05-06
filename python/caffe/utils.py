import os,sys,logging
try:
    caffe_root = '/home/stive/workspace/caffe_train/'
    sys.path.insert(0, caffe_root + 'python')
    import caffe
except ImportError:
    logging.fatal("Cannot find caffe!")
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import math


def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in range(0, num):
        ret.append(var[0])
    else:
      for i in range(0, num):
        ret.append(var)
    return ret

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, group=1, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    bn_eps=0.001, bn_moving_avg_fraction=0.999, Use_DeConv = False, use_global_stats = False,
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='msra'),
        #'weight_filler': dict(type='msra', std=0.01),
        'bias_term': False,
        }
    eps = bn_params.get('eps', bn_eps)
    moving_average_fraction = bn_params.get('moving_average_fraction', bn_moving_avg_fraction)
    use_global_stats = bn_params.get('use_global_stats', use_global_stats)
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
        }
    bn_lr_mult = lr_mult
    if use_global_stats:
      # only specify if use_global_stats is explicitly provided;
      # otherwise, use_global_stats_ = this->phase_ == TEST;
      bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': eps,
          'use_global_stats': use_global_stats,
          }
      # not updating scale/bias parameters
      bn_lr_mult = 0
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult * 2, decay_mult=0)],
          'filler': dict(value=1.0),
          'bias_filler': dict(value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='msra'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    if Use_DeConv:
        net[conv_name] = L.Deconvolution(net[from_layer], param=[dict(lr_mult=0, decay_mult=0)],
                                         convolution_param=dict(
                                         bias_term=False, num_output=num_output, kernel_size=kernel_h,
                                         stride=stride_h, pad=pad_h, weight_filler=dict(type="msra")))
    else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
            kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
            kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
            stride_h=stride_h, stride_w=stride_w, **kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if group > 1:
      net.update(conv_name, {'group': group})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu6'.format(conv_name)
    net[relu_name] = L.ReLU6(net[conv_name], in_place=True)


def round_filters(filters, width_coefficient, min_depth=None, depth_divisor= 8, skip=False):
  """Round number of filters based on depth multiplier."""
  orig_f = filters
  if skip or not width_coefficient:
    return filters

  filters *= width_coefficient
  min_depth = min_depth or depth_divisor
  new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += depth_divisor
  logging.info('round_filter input=%s output=%s', orig_f, new_filters)
  return int(new_filters)


def round_repeats(repeats, depth_coefficient, skip=False):
  """Round number of filters based on depth multiplier."""
  if skip or not depth_coefficient:
    return repeats
  return int(math.ceil(depth_coefficient * repeats))
