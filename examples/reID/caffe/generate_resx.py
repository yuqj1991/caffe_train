import _init_paths
import os
import os.path as osp
import caffe
from caffe import layers as L, params as P
from caffe import tools
from caffe.model_libs import *

def res_unit(net, bottom, in_c, out_c, stride, base_name, post, is_train=False):

  assert (out_c % 4 == 0)
  param = [dict(lr_mult=0.1,decay_mult=1), dict(lr_mult=0.2,decay_mult=0)]

  pase_name = base_name
  base_name = base_name + post
  if (in_c != out_c):
    #param = pase_name + '_branch1'
    net['res'+base_name+'_branch1'], net['bn'+base_name+'_branch1'], net['scale'+base_name+'_branch1'] = \
      conv_bn_scale(bottom,  1, out_c, stride=stride, is_train=is_train, has_relu=False, bias_term=False, param=param)
      #conv_bn_scale(bottom,  1, out_c, base_param_name=param, stride=stride, is_train=is_train, has_relu=False, bias_term=False)
    identity = net['scale'+base_name+'_branch1']
  else:
    identity = bottom

  #param = pase_name + '_branch2a'
  net['res'+base_name+'_branch2a'], net['bn'+base_name+'_branch2a'], net['scale'+base_name+'_branch2a'], net['res'+base_name+'_branch2a_relu'] = \
      conv_bn_scale(bottom,  1, out_c/4, stride=stride, is_train=is_train, has_relu=True, bias_term=False, param=param)
      #conv_bn_scale(bottom,  1, out_c/4, base_param_name=param, stride=stride, is_train=is_train, has_relu=True, bias_term=False)

  #param = pase_name + '_branch2b'
  net['res'+base_name+'_branch2b'], net['bn'+base_name+'_branch2b'], net['scale'+base_name+'_branch2b'], net['res'+base_name+'_branch2b_relu'] = \
      conv_bn_scale(net['res'+base_name+'_branch2a_relu'],  3, out_c/4, pad=1, is_train=is_train, has_relu=True, bias_term=False, param=param)
      #conv_bn_scale(net['res'+base_name+'_branch2a_relu'],  3, out_c/4, base_param_name=param, pad=1, is_train=is_train, has_relu=True, bias_term=False)

  #param = pase_name + '_branch2c'
  net['res'+base_name+'_branch2c'], net['bn'+base_name+'_branch2c'], net['scale'+base_name+'_branch2c'] = \
      conv_bn_scale(net['res'+base_name+'_branch2b_relu'],  1, out_c, is_train=is_train, has_relu=False, bias_term=False, param=param)
      #conv_bn_scale(net['res'+base_name+'_branch2b_relu'],  1, out_c, base_param_name=param, is_train=is_train, has_relu=False, bias_term=False)

  final = net['scale'+base_name+'_branch2c']

  net['res'+base_name] = L.Eltwise(identity, final)
  net['res'+base_name+'_relu'] = L.ReLU(net['res'+base_name], in_place=True)
  final_name = 'res'+base_name+'_relu'
  return net, final_name
      
def res50_body(net, data, post, is_train):
  net['conv1'+post], net['bn_conv1'+post], net['scale_conv1'+post], net['conv1_relu'+post] = \
      conv_bn_scale(net[data],       7, 64, pad = 3, stride = 2, is_train=is_train, has_relu=True, param = [dict(lr_mult=0.01,decay_mult=1), dict(lr_mult=0.02,decay_mult=0)])

  net['pool1'+post] = max_pool(net['conv1_relu'+post], 3, stride=2)
  names, outs = ['2a', '2b', '2c'], [256, 256, 256]
  pre_out = 64
  final = 'pool1'+post
  for (name, out) in zip(names, outs):
    net, final = res_unit(net, net[final], pre_out, out, 1, name, post, is_train=is_train)
    pre_out = out

  names, outs = ['3a', '3b', '3c', '3d'], [512, 512, 512, 512]
  for (name, out) in zip(names, outs):
    if (name == '3a'):
      net, final = res_unit(net, net[final], pre_out, out, 2, name, post, is_train=is_train)
    else:
      net, final = res_unit(net, net[final], pre_out, out, 1, name, post, is_train=is_train)
    pre_out = out

  names = ['4a', '4b', '4c', '4d', '4e', '4f']
  out = 1024
  for name in names:
    if (name == '4a'):
      net, final = res_unit(net, net[final], pre_out, out, 2, name, post, is_train=is_train)
    else:
      net, final = res_unit(net, net[final], pre_out, out, 1, name, post, is_train=is_train)
    pre_out = out

  names = ['5a', '5b', '5c']
  out = 2048
  for name in names:
    if (name == '5a'):
      net, final = res_unit(net, net[final], pre_out, out, 2, name, post, is_train=is_train)
    else:
      net, final = res_unit(net, net[final], pre_out, out, 1, name, post, is_train=is_train)
    pre_out = out

  net['pool5'+post] = ave_pool(net[final], 7, 1)
  final = 'pool5'+post

  return net, final
  
# main netspec wrapper
def res50_train(mean_value, list_file, is_train, batch_size):
  # setup the python data layer 
  net = caffe.NetSpec()
  net.data, net.label \
                  = L.ReidData(transform_param=dict(mirror=True,crop_size=224,mean_value=mean_value), 
           reid_data_param=dict(source=list_file,batch_size=batch_size, new_height=256, new_width=256,
              pos_fraction=1,neg_fraction=1,pos_limit=1,neg_limit=4,pos_factor=1, neg_factor=1.01), 
           ntop = 2)
  
  net, final = res50_body(net, 'data',   '', is_train)

  param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)]
  net['score']     = fc_relu(net[final],       nout=751, is_train=is_train, has_relu=False, param=param)
  #net['euclidean'], net['label_dif'] = L.PairEuclidean(net[final], net['label'], ntop = 2)
  net['label_dif'] = L.PairReidLabel(net['label'], propagate_down=[0], ntop = 1)

  net['feature_a'], net['feature_b'] = L.Slice(net[final], slice_param=dict(axis=0, slice_point=batch_size), ntop = 2)
  net['euclidean'] = L.Eltwise(net['feature_a'], net['feature_b'], operation = P.Eltwise.PROD)
  net['score_dif'] = fc_relu(net['euclidean'], nout=2,   is_train=is_train, has_relu=False, param=param)
  
  net['loss']      = L.SoftmaxWithLoss(net['score'],     net['label']    , propagate_down=[1,0], loss_weight=0.5)
  net['loss_dif']  = L.SoftmaxWithLoss(net['score_dif'], net['label_dif'], propagate_down=[1,0], loss_weight=  1)
  return str(net.to_proto())

def res50_score(input_param = dict(shape=dict(dim=[1, 3, 224, 224]))):
  # setup the python data layer 
  net = caffe.NetSpec()
  net['data']       = L.Input(input_param = input_param)
  net, final        = res50_body(net, 'data', '', is_train=False)
  net['score']      = fc_relu(net[final],     nout=751, is_train=False, has_relu=False)
  net['prediction'] = L.Softmax(net['score'])
  return str(net.to_proto())

workdir = osp.join(osp.dirname(__file__), 'resxx50')
if not os.path.isdir(workdir):
  os.makedirs(workdir)

logdir = osp.join(workdir, 'log')
if not os.path.isdir(logdir):
  os.makedirs(logdir)

snapshotdir = osp.join(workdir, 'snapshot')
if not os.path.isdir(snapshotdir):
  os.makedirs(snapshotdir)
print('Work Dir : {}'.format(workdir))

train_proto = osp.join(workdir, "train.proto")

solverproto = tools.CaffeSolver(trainnet_prototxt_path = train_proto, testnet_prototxt_path = None)
solverproto.sp['display'] = "100"
solverproto.sp['base_lr'] = "0.001"
solverproto.sp['stepsize'] = "16000"
solverproto.sp['max_iter'] = "18000"
solverproto.sp['snapshot'] = "1000"
solverproto.sp['iter_size'] = "2"
solverproto.sp['snapshot_prefix'] = "\"{}/snapshot/resxx50.full\"".format(workdir)
solverproto.write(osp.join(workdir, 'solver.proto'))
batch_size = 16

list_file = 'examples/market1501/lists/train.lst'

mean_value = [97.8286, 99.0468, 105.606]
# write train net.
with open(train_proto, 'w') as f:
  f.write(res50_train(mean_value, list_file, True, batch_size))

dev_proto = osp.join(workdir, "dev.proto")
with open(dev_proto, 'w') as f:
  f.write(res50_score())

train_shell = osp.join(workdir, "train.sh")
with open(train_shell, 'w') as f:          
  f.write('#!/usr/bin/env sh\n')           
  f.write('model_dir={}\n'.format(workdir))                           
  f.write('pre_train_dir=${HOME}/datasets/model_pretrained/resnet\n\n')       
  f.write('GLOG_log_dir=${model_dir}/log ./build/tools/caffe train ')      
  f.write('--solver ${model_dir}/solver.proto --weights ${pre_train_dir}/ResNet-50-model.caffemodel $@')
