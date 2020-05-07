from __future__ import division
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
from typing import Text, Tuple, Union
import functools

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

def ConvBNLayer(net, from_layer, out_layer, use_bn, num_output,
    kernel_size, pad, stride, group=1, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    bn_eps=0.001, bn_moving_avg_fraction=0.999, Use_DeConv = False, use_global_stats = False,
    use_relu = False, use_swish= False,
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
  if use_swish:
      swish_name = '{}_swish'.format(conv_name)
      net[swish_name] = L.Swish(net[conv_name], in_place=True)


def SeparableConv(net, from_layer, use_bn, num_output, channel_multplutir, 
    kernel_size, pad, stride, group=1, dilation=1, use_scale=True, lr_mult=1,
    bn_eps=0.001, bn_moving_avg_fraction=0.999, use_global_stats = False,
    use_relu = False, use_swish= False, layerPrefix = "", **bn_params):
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  assert num_output * channel_multplutir == group
  out_layer = "{}_Deconv_{}x{}".format(layerPrefix, kernel_h, kernel_w)  # kernel 3 x 3
  ConvBNLayer(net, from_layer, out_layer, use_bn, num_output * channel_multplutir, 
                  kernel_size, pad, stride,use_relu= use_relu, use_swish= use_swish,
                  group=group, dilation=dilation, use_scale=use_scale, lr_mult=1, use_global_stats = use_global_stats)
  point_layer = "{}_point_conv_{}x{}".format(layerPrefix, 1, 1) # kernel 1 x 1
  ConvBNLayer(net, out_layer, point_layer,  use_bn, num_output, kernel_size = 1, pad= 0, use_relu= use_relu, use_swish= use_swish,
                    stride = 1, group=1, dilation=1, use_scale=use_scale, lr_mult=1, use_global_stats = use_global_stats)
  out_layer = point_layer
  return out_layer


def NormalConv(net, from_layer, use_bn, num_output, channel_multplutir, 
    kernel_size, pad, stride, group=1, dilation=1, use_scale=True, lr_mult=1,
    bn_eps=0.001, bn_moving_avg_fraction=0.999, use_global_stats = False,
    use_relu = False, use_swish= False, layerPrefix = "", **bn_params):
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  assert num_output * channel_multplutir == group
  out_layer = "{}_Conv_{}x{}".format(layerPrefix, kernel_h, kernel_w)  # kernel 3 x 3
  ConvBNLayer(net, from_layer, out_layer, use_bn, num_output * channel_multplutir, 
                  kernel_size, pad, stride,use_relu= use_relu, use_swish= use_swish,
                  group=group, dilation=dilation, use_scale=use_scale, lr_mult=1, use_global_stats = use_global_stats)
  return out_layer


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


################################################################################
def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
  """Parse the image size and return (height, width).

  Args:
    image_size: A integer, a tuple (H, W), or a string with HxW format.

  Returns:
    A tuple of integer (height, width).
  """
  if isinstance(image_size, int):
    # image_size is integer, with the same width and height.
    return (image_size, image_size)

  if isinstance(image_size, str):
    # image_size is a string with format WxH
    width, height = image_size.lower().split('x')
    return (int(height), int(width))

  if isinstance(image_size, tuple):
    return image_size

  raise ValueError('image_size must be an int, WxH string, or (height, width)'
                   'tuple. Was %r' % image_size)


def get_feat_sizes(image_size: Union[Text, int, Tuple[int, int]],
                   max_level: int):
  """Get feat widths and heights for all levels.

  Args:
    image_size: A integer, a tuple (H, W), or a string with HxW format.
    max_level: maximum feature level.

  Returns:
    feat_sizes: a list of tuples (height, width) for each level.
  """
  image_size = parse_image_size(image_size)
  feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
  feat_size = image_size
  for _ in range(1, max_level + 1):
    feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
    feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
  return feat_sizes


################################################################################
def verify_feats_size(net, feats, feat_sizes, min_level, max_level):
  """Verify the feature map sizes."""

  expected_output_size = feat_sizes[min_level:max_level + 1]
  for cnt, size in enumerate(expected_output_size):
    h_id, w_id = (2, 3)
    if net.blobs[feats[cnt]].shape[h_id] != size['height']:
      raise ValueError(
          'feats[{}] has shape {} but its height should be {}.'
          '(input_height: {}, min_level: {}, max_level: {}.)'.format(
              cnt, net.blobs[feats[cnt]].shape, size['height'], feat_sizes[0]['height'],
              min_level, max_level))
    if net.blobs[feats[cnt]].shape[w_id] != size['width']:
      raise ValueError(
          'feats[{}] has shape {} but its width should be {}.'
          '(input_width: {}, min_level: {}, max_level: {}.)'.format(
              cnt, net.blobs[feats[cnt]].shape, size['width'], feat_sizes[0]['width'],
              min_level, max_level))


################################################################################
def resample_feature_map(net, from_layer, use_global_stats,
                          target_height, target_width, target_channels,
                          use_relu = False, use_swish= True,
                          current_height = None, current_width = None, 
                          current_channels = None, layerPrefix = '',
                          apply_bn=False, is_training=None, conv_after_downsample=False, 
                          use_nearest_resize=False, pooling_type=None):
  """Resample input feature map to have target number of channels and size."""
  
  if current_height is None or current_width is None or current_channels is None:
      raise ValueError( 'shape[1](heigit: {}) or shape[2](width: {}) or shape[3](channels: {}) of feat is None'.
                                                                          format(current_height, current_width, current_channels))
  if apply_bn and is_training is None:
      raise ValueError('If BN is applied, need to provide is_training')

  def _maybe_apply_1x1(net, from_layer):
      """Apply 1x1 conv to change layer width if necessary."""
      out_layer = from_layer
      if current_channels != target_channels:
        out_layer= "{}_Conv_1x1_project_{}_{}".format(layerPrefix, current_channels, target_channels)
        ConvBNLayer(net, from_layer, out_layer, use_bn = apply_bn, use_relu = False, use_swish= False,
                    num_output= target_channels, kernel_size= 1, pad= 0, stride= 1,
                    lr_mult=1, use_scale=apply_bn, use_global_stats= use_global_stats)
      return out_layer

  # If conv_after_downsample is True, when downsampling, apply 1x1 after
  # downsampling for efficiency.
  out_layer = from_layer
  if current_height > target_height and current_width > target_width:
    if not conv_after_downsample:
      out_layer = _maybe_apply_1x1(net, out_layer)
    height_stride = int((current_height - 1) // target_height + 1)
    width_stride = int((current_width - 1) // target_width + 1)
    if pooling_type == 'max' or pooling_type is None:
      Global_poolingName= "{}_Pool".format(layerPrefix)
      net[Global_poolingName] = L.Pooling(net[out_layer], pool=P.Pooling.MAX, kernel_h = height_stride + 1, kernel_w= width_stride + 1,
                                          stride_h=height_stride, stride_w=width_stride, pad = 1,
                                          global_pooling=False)
      out_layer = Global_poolingName
    elif pooling_type == 'avg':
      Global_poolingName= "{}_Pool".format(layerPrefix)
      net[Global_poolingName] = L.Pooling(net[out_layer], pool=P.Pooling.AVG, kernel_h = height_stride + 1, kernel_w= width_stride + 1,
                                          stride_h=height_stride, stride_w=width_stride, pad = 1,
                                          global_pooling=False)
      out_layer = Global_poolingName
    else:
      raise ValueError('Unknown pooling type: {}'.format(pooling_type))
    if conv_after_downsample:
      out_layer = _maybe_apply_1x1(net, out_layer)
  elif current_height <= target_height and current_width <= target_width:
    # Upsampling layer
    out_layer = _maybe_apply_1x1(net, out_layer)
    if current_height < target_height or current_width < target_width:
      height_scale = target_height // current_height
      width_scale = target_width // current_width
      assert height_scale == width_scale
      if (use_nearest_resize or target_height % current_height != 0 or
          target_width % current_width != 0):
        Upsample_name = "{}_bilinear_Upsample".format(layerPrefix)
        net[Upsample_name] = L.Upsample(net[out_layer], scale= height_scale)
        out_layer = Upsample_name
      else:
        Upsample_name = "{}_Deconv_Upsample".format(layerPrefix)
        ConvBNLayer(net, out_layer, Upsample_name, use_bn= apply_bn, use_relu = use_relu, use_swish= use_swish,
          num_output= target_channels, kernel_size= height_scale + 1, pad= 1, stride= height_scale,
          lr_mult=1, Use_DeConv= True, use_scale= apply_bn, use_global_stats= use_global_stats)
        out_layer = Upsample_name
  else:
    raise ValueError(
        'Incompatible target feature map size: target_height: {},'
        'target_width: {}'.format(target_height, target_width))
  return out_layer

################################################################################
def BuildBiFPNLayer(net, feats, feat_sizes, fpn_nodes, layerPrefix = '', fpn_out_filters= 88, min_level = 3, max_level = 7, 
                    use_global_stats = True, use_relu = False, use_swish= True,  concat_method= "fast_attention",
                    con_bn_act_pattern = False,
                    apply_bn=True, is_training=True, conv_after_downsample=False, separable_conv = True,
                    use_nearest_resize=False, pooling_type= None):
  """Builds a feature pyramid given previous feature pyramid and config."""
  for i, fnode in enumerate(fpn_nodes):
    new_node_height = feat_sizes[fnode['feat_level']]['height']
    new_node_width = feat_sizes[fnode['feat_level']]['width']
    nodes = []
    for idx, input_offset in enumerate(fnode['inputs_offsets']):
      input_node = feats[input_offset]
      input_node = resample_feature_map(net, from_layer= input_node, use_global_stats= use_global_stats, use_relu= use_relu, 
                          target_height= new_node_height, target_width= new_node_width, target_channels= fpn_out_filters,
                          current_height = None, current_width = None, 
                          current_channels = None, layerPrefix = '{}_{}_{}'.format(idx, input_offset, len(feats)),
                          apply_bn= apply_bn, is_training= is_training, conv_after_downsample= conv_after_downsample, 
                          use_nearest_resize= use_nearest_resize, pooling_type= pooling_type)
      nodes.append(input_node)
    
    # Combine all nodes.
    if concat_method == "fast_attention":
      Attention_Name = "{}_concat_fast_attention".format(layerPrefix)
      net[Attention_Name] = L.WightEltwise(*nodes, wighted_eltwise_param= dict(operation= P.WightedEltwise.FASTER, 
                                                                                        weight_filler=dict(type="msra")))
    elif concat_method == "softmax_attention":
      Attention_Name = "{}_concat_softmax_attention".format(layerPrefix)
      net[Attention_Name] = L.WightEltwise(*nodes, wighted_eltwise_param= dict(operation= P.WightedEltwise.SOFTMAX, 
                                                                                        weight_filler=dict(type="msra")))        
    elif concat_method == "sum_attention":
      Attention_Name = "{}_concat_sum_attention".format(layerPrefix)
      net[Attention_Name] = L.WightEltwise(*nodes, wighted_eltwise_param= dict(operation= P.WightedEltwise.FASTER, 
                                                                                        weight_filler=dict(type="msra")))
    else:
      raise ValueError('unknown weight_method {}'.format(concat_method))
    # operation after combine, like conv & bn
    out_layer = Attention_Name
    if not con_bn_act_pattern:
      Swish_Name = "{}_swish".format(layerPrefix)
      net[Swish_Name] = L.Swish(net[out_layer], in_place = True)
      out_layer = Swish_Name
    if separable_conv: # need batch-norm
      Deconv_Name = "{}_Deconv_3x3".format(layerPrefix)
      ConvBNLayer(net, out_layer, Deconv_Name, use_bn = apply_bn, use_relu = False, use_swish= False,
                        num_output= fpn_out_filters, kernel_size= 3, pad= 1, stride= 1,
                        lr_mult=1, use_scale=apply_bn, use_global_stats= use_global_stats, Use_DeConv= True)
      out_layer = Deconv_Name
      Point_Name = "{}_conv_1x1_point".format(layerPrefix)
      ConvBNLayer(net, out_layer, Point_Name, use_bn = apply_bn, use_relu = False, use_swish= False,
                    num_output= fpn_out_filters, kernel_size= 1, pad= 0, stride= 1,
                    lr_mult=1, use_scale=apply_bn, use_global_stats= use_global_stats, Use_DeConv= False)
      out_layer = Point_Name
    else:
      Conv_name = "{}_conv_3x3".format(layerPrefix)
      ConvBNLayer(net, out_layer, out_layer, use_bn = apply_bn, use_relu = False, use_swish= False,
                    num_output= fpn_out_filters, kernel_size= 3, pad= 1, stride= 1,
                    lr_mult=1, use_scale=apply_bn, use_global_stats= use_global_stats, Use_DeConv= False)
      out_layer = Conv_name
    feats.append(out_layer)
  
  output_feats = {}
  for l in range(min_level, max_level + 1):
    for i, fnode in enumerate(reversed(fpn_nodes)):
      if fnode['feat_level'] == l:
        output_feats[l] = feats[-1 - i]
        break
  return output_feats


###############################################################################
def class_net(net, images,
              num_classes, num_anchors,
              num_filters, is_training,
              layerPrefix = "",
              separable_conv=True, repeats=4):
  """Class prediction network."""
  out_layer = "{}_Class_Conv".format(layerPrefix)
  if separable_conv:
    conv_op = functools.partial(SeparableConv, net= net, group= num_filters, channel_multplutir = 1, layerPrefix = "{}_DepthWise_Conv".format(layerPrefix))
  else:
    conv_op = functools.partial(NormalConv, net= net, out_layer= out_layer, layerPrefix = "{}_Normal_Conv".format(layerPrefix))
  for _ in range(repeats):
    images = conv_op
    images = conv_op(
        from_layer= images,
        num_output= num_filters,
        kernel_size = 3, lr_mult= 1, use_bn= True, use_scale= True, use_swish= True, 
        layerPrefix = "{}_Conv".format(layerPrefix), use_global_stats= is_training)

  classes = conv_op(
        from_layer= images,
        num_output= num_classes * num_anchors,
        kernel_size = 3, lr_mult= 1, use_bn= False, use_scale= False, use_swish= False, 
        layerPrefix = "{}_Class_Predict".format(layerPrefix), use_global_stats= is_training)
  return classes


def box_net(net, images, num_anchors, num_filters,
            is_training, repeats=4,
            separable_conv=True, layerPrefix=''):
  """Box regression network."""
  out_layer = "{}_Class_Conv".format(layerPrefix)
  if separable_conv:
    conv_op = functools.partial(SeparableConv, net= net, group= num_filters, channel_multplutir = 1)
  else:
    conv_op = functools.partial(NormalConv, net= net, out_layer= out_layer)
  for _ in range(repeats):
    images = conv_op
    images = conv_op(
        from_layer= images,
        num_output= num_filters,
        kernel_size = 3, lr_mult= 1, use_bn= True, use_scale= True, use_swish= True, 
        use_global_stats= is_training, layerPrefix = "{}_Conv".format(layerPrefix))

  boxes = conv_op(
        from_layer= images,
        num_output= 4 * num_anchors,
        kernel_size = 3, lr_mult= 1, use_bn= False, use_scale= False, use_swish= False, 
        use_global_stats= is_training, layerPrefix = "{}_Box_Predict".format(layerPrefix))
  return boxes


def Build_class_and_box_outputs(net, feats, fpn_num_filters, num_classes, min_level, max_level, is_training_bn, 
                                  aspect_ratios= [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)], num_scales = 4):
  """Builds box net and class net.

  Args:
   feats: input tensor.
  Returns:
   A tuple (class_outputs, box_outputs) for class/box predictions.
  """

  class_outputs = {}
  box_outputs = {}
  num_anchors = len(aspect_ratios) * num_scales
  cls_fsize = fpn_num_filters
  for level in range(min_level, max_level + 1):
    class_outputs[level] = class_net(
        net= net, 
        images=feats[level],
        num_classes=num_classes,
        num_anchors=num_anchors, num_filters=cls_fsize,
        is_training= is_training_bn, repeats=3, separable_conv=True, layerPrefix= "{}_Class".format(level))

  box_fsize = fpn_num_filters
  for level in range(min_level, max_level + 1):
    box_outputs[level] = box_net(
        net= net, 
        images=feats[level],
        num_anchors=num_anchors,
        num_filters=box_fsize,
        is_training=is_training_bn,
        repeats=3, separable_conv=True, layerPrefix= "{}_Box".format(level))

  return class_outputs, box_outputs