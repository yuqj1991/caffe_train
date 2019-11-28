import _init_paths
import os
import os.path as osp
import caffe
from caffe import layers as L, params as P
from caffe import tools
from caffe.model_libs import *

def caffenet_body(net, data, post, is_train):
  # the net itself
  net['conv1'+post], net['relu1'+post] = conv_relu(net[data],         11, 96, stride=4, is_train=is_train)
  net['pool1'+post]  = max_pool(net['relu1'+post], 3, stride=2)
  net['norm1'+post]  = L.LRN(net['pool1'+post], local_size=5, alpha=1e-4, beta=0.75, engine=P.LRN.CAFFE)
  net['conv2'+post], net['relu2'+post] = conv_relu(net['norm1'+post], 5, 256, pad=2, group=2, is_train=is_train)
  net['pool2'+post]  = max_pool(net['relu2'+post], 3, stride=2)
  net['norm2'+post]  = L.LRN(net['pool2'+post], local_size=5, alpha=1e-4, beta=0.75, engine=P.LRN.CAFFE)
  net['conv3'+post], net['relu3'+post] = conv_relu(net['norm2'+post], 3, 384, pad=1, is_train=is_train)
  net['conv4'+post], net['relu4'+post] = conv_relu(net['relu3'+post], 3, 384, pad=1, group=2, is_train=is_train)
  net['conv5'+post], net['relu5'+post] = conv_relu(net['relu4'+post], 3, 256, pad=1, group=2, is_train=is_train)
  net['pool5'+post]  = max_pool(net['relu5'+post], 3, stride=2)
  net['fc6'+post], net['relu6'+post]   = fc_relu(net['pool5'+post], 4096, is_train=is_train)
  net['drop6'+post]  = L.Dropout(net['relu6'+post], in_place=True)
  net['fc7'+post], net['relu7'+post]   = fc_relu(net['drop6'+post], 4096, is_train=is_train)
  net['drop7'+post]  = L.Dropout(net['relu7'+post], in_place=True)
  #n.score = L.InnerProduct(n.drop7, num_output=20, weight_filler=dict(type='gaussian', std=0.01))
  #n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
  final = 'drop7'+post
  return net, final
  
# main netspec wrapper
def caffenet_train(mean_value, list_file, is_train=True):
  # setup the python data layer 
  net = caffe.NetSpec()
  net.data, net.label \
    = L.ReidData(transform_param=dict(mirror=True,crop_size=227,mean_value=mean_value), 
                reid_data_param=dict(source=list_file,batch_size=128,new_height=256,new_width=256,
                pos_fraction=1,neg_fraction=1,pos_limit=1,neg_limit=4,pos_factor=1,neg_factor=1.01), 
                ntop = 2)
  
  net, final = caffenet_body(net, 'data', '', is_train)

  net['score']     = fc_relu(net[final],     nout=751, is_train=is_train, has_relu=False)
  net['euclidean'], net['label_dif'] = L.PairEuclidean(net[final], net['label'], ntop = 2)
  net['score_dif'] = fc_relu(net['euclidean'], nout=2, is_train=is_train, has_relu=False)
  
  net['loss']      = L.SoftmaxWithLoss(net['score'],     net['label']    , propagate_down=[1,0], loss_weight=1)
  net['loss_dif']  = L.SoftmaxWithLoss(net['score_dif'], net['label_dif'], propagate_down=[1,0], loss_weight=0.5)
  return str(net.to_proto())

def caffenet_dev(data_param = dict(shape=dict(dim=[2, 3, 227, 227])), label_param = dict(shape=dict(dim=[2]))):
  # setup the python data layer 
  net = caffe.NetSpec()
  net['data']  = L.Input(input_param = data_param)
  net['label'] = L.Input(input_param = label_param)
  net, final   = caffenet_body(net, 'data', '', is_train=False)
  net['score']     = fc_relu(net[final],     nout=751, is_train=False, has_relu=False)
  net['euclidean'], net['label_dif'] = L.PairEuclidean(net[final], net['label'], ntop = 2)
  net['score_dif'] = fc_relu(net['euclidean'], nout=2,   is_train=False, has_relu=False)
  return str(net.to_proto())

def caffenet_score(input_param = dict(shape=dict(dim=[1, 3, 227, 227]))):
  # setup the python data layer 
  net = caffe.NetSpec()
  net['data']       = L.Input(input_param = input_param)
  net, final = caffenet_body(net, 'data', '', is_train=False)
  net['score']      = fc_relu(net[final],     nout=751, is_train=False, has_relu=False)
  net['prediction'] = L.Softmax(net['score'])
  return str(net.to_proto())

workdir = osp.join(osp.dirname(__file__), 'caffenet')
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
solverproto.sp['display'] = "20"
solverproto.sp['base_lr'] = "0.001"
solverproto.sp['stepsize'] = "16000"
solverproto.sp['max_iter'] = "18000"
solverproto.sp['snapshot'] = "2000"
solverproto.sp['snapshot_prefix'] = "\"{}/snapshot/caffenet.full\"".format(workdir)
solverproto.write(osp.join(workdir, 'solver.proto'))

list_file = 'examples/market1501/lists/train.lst'

mean_value = [97.8286, 99.0468, 105.606]

# write train net.
with open(train_proto, 'w') as f:
  f.write(caffenet_train(mean_value, list_file, True))

dev_proto = osp.join(workdir, "dev.proto")
with open(dev_proto, 'w') as f:
  f.write(caffenet_score())

dep_proto = osp.join(workdir, "deploy.proto")
with open(dep_proto, 'w') as f:
  f.write(caffenet_dev())

train_shell = osp.join(workdir, "train.sh")
with open(train_shell, 'w') as f:          
  f.write('#!/usr/bin/env sh\n')           
  f.write('model_dir=models/market1501/caffenet\n')            
  f.write('pre_train_dir=${HOME}/datasets/model_pretrained/caffenet\n\n')       
  f.write('GLOG_log_dir=${model_dir}/log ./build/tools/caffe train ')      
  f.write('--solver ${model_dir}/solver.proto --weights ${pre_train_dir}/bvlc_reference_caffenet.caffemodel $@')  
