import _init_paths
import os
import os.path as osp
import caffe
from caffe import layers as L, params as P
from caffe import tools
from caffe.model_libs import *

def vgg16_body(net, data, post, is_train):
  # the net itself
  #param = [dict(lr_mult=0.5,decay_mult=1), dict(lr_mult=1,decay_mult=0)]
  param = None
  nunits_list = [2, 2, 3, 3, 3]
  nouts = [64, 128, 256, 512, 512]
  main_name = data
  for idx, (nout, nunits) in enumerate(zip(nouts, nunits_list)): # for each depth and nunits
    for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
      convn = 'conv' + str(idx+1) + '_' + str(unit) + post
      relun = 'relu' + str(idx+1) + '_' + str(unit) + post
      #param = 'conv' + str(idx+1) + '_' + str(unit) + 'w'
      net[convn], net[relun] = conv_relu(net[main_name], 3, nout, pad = 1, is_train=is_train, param=param)
      main_name = relun
    pooln = 'pool' + str(idx+1) + post
    net[pooln] = L.Pooling(net[main_name], stride = 2, kernel_size = 2, pool = P.Pooling.MAX, ntop=1)
    main_name = pooln

  net['fc6'+post], net['relu6'+post]   = fc_relu(net[main_name], 4096,    is_train=is_train, param = param)
  net['drop6'+post]                    = L.Dropout(net['relu6'+post], in_place=True, dropout_ratio=0.5)
  net['fc7'+post], net['relu7'+post]   = fc_relu(net['drop6'+post], 4096, is_train=is_train, param = param)
  net['drop7'+post]                    = L.Dropout(net['relu7'+post], in_place=True, dropout_ratio=0.5)
  final_name = 'drop7'+post
  return net, final_name
  
# main netspec wrapper
def vgg16_train(mean_value, list_file, is_train=True):
  # setup the python data layer 
  net = caffe.NetSpec()
  net.data, net.label \
                  = L.ReidData(transform_param=dict(mirror=True,crop_size=224,mean_value=mean_value), 
                               reid_data_param=dict(source=list_file,batch_size=32,new_height=256,new_width=256,
                               pos_fraction=1,neg_fraction=1,pos_limit=1,neg_limit=4,pos_factor=1,neg_factor=1.01), 
                               ntop = 2)
  
  net, final = vgg16_body(net,   'data',   '', is_train)

  #param            = param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)]
  param            = None

  net['score']     = fc_relu(net[final],     nout=751,   is_train=is_train, has_relu=False, param = param)

  net['euclidean'], net['label_dif'] = L.PairEuclidean(net[final], net['label'], ntop=2)
  net['score_dif'] = fc_relu(net['euclidean'], nout=2,   is_train=is_train, has_relu=False, param = param)
  
  net['loss']      = L.SoftmaxWithLoss(net['score'],     net['label']    , propagate_down=[1,0], loss_weight=  1)
  net['loss_dif']  = L.SoftmaxWithLoss(net['score_dif'], net['label_dif'], propagate_down=[1,0], loss_weight=0.5)
  return str(net.to_proto())

def vgg16_dev(data_param = dict(shape=dict(dim=[2, 3, 224, 224])), label_param = dict(shape=dict(dim=[2]))):
  # setup the python data layer 
  net = caffe.NetSpec()
  net['data']   = L.Input(input_param = data_param)
  net['label']  = L.Input(input_param = label_param)
  net, final    = vgg16_body(net,   'data',   '', False)
  net['score']     = fc_relu(net[final],       nout=751, is_train=False, has_relu=False)
  net['euclidean'], net['label_dif'] = L.PairEuclidean(net[final], net['label'], ntop = 2)
  net['score_dif'] = fc_relu(net['euclidean'], nout=2,   is_train=False, has_relu=False)
  return str(net.to_proto())

def vgg16_score(input_param = dict(shape=dict(dim=[1, 3, 224, 224]))):
  # setup the python data layer 
  net = caffe.NetSpec()
  net['data']       = L.Input(input_param = input_param)
  net, final        = vgg16_body(net, 'data', '', is_train=False)
  net['score']      = fc_relu(net[final],     nout=751, is_train=False, has_relu=False)
  net['prediction'] = L.Softmax(net['score'])
  return str(net.to_proto())

workdir = osp.join(osp.dirname(__file__), 'vgg16')
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
solverproto.sp['display'] = "50"
solverproto.sp['base_lr'] = "0.001"
solverproto.sp['stepsize'] = "16000"
solverproto.sp['max_iter'] = "18000"
solverproto.sp['snapshot'] = "2000"
solverproto.sp['iter_size'] = "1"
solverproto.sp['type'] = "\"Nesterov\""
solverproto.sp['snapshot_prefix'] = "\"{}/snapshot/vgg16.full\"".format(workdir)
solverproto.write(osp.join(workdir, 'solver.proto'))

list_file = 'examples/market1501/lists/train.lst'

mean_value = [97.8286, 99.0468, 105.606]
# write train net.
with open(train_proto, 'w') as f:
  f.write(vgg16_train(mean_value, list_file, True))

dev_proto = osp.join(workdir, "dev.proto")
with open(dev_proto, 'w') as f:
  f.write(vgg16_score())

dep_proto = osp.join(workdir, "deploy.proto")
with open(dep_proto, 'w') as f:
  f.write(vgg16_dev())

train_shell = osp.join(workdir, "train.sh")
with open(train_shell, 'w') as f:
  f.write('#!/usr/bin/env sh\n')
  f.write('model_dir=models/market1501/vgg16\n')
  f.write('pre_train_dir=${HOME}/datasets/model_pretrained/vgg\n\n')
  f.write('GLOG_log_dir=${model_dir}/log ./build/tools/caffe train ')
  f.write('--solver ${model_dir}/solver.proto --weights ${pre_train_dir}/VGG_ILSVRC_16_layers.caffemodel $@')
