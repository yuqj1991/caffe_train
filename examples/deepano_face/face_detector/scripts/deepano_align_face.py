import numpy
import os, sys
import caffe
from caffe import layers as L, params as P

def deepano_conv_relu(net,conv_name,fileter_kernel_size,padding,striding,num_output,
                      weight_lr_mult, weight_decay_name, bias_lr_mult, bias_decay_mult,
                      weight_filler_type, bias_filler_type):
    net. = L.Convolution(net.data,param=[{"lr_mult": weight_lr_mult, "decay_mult": weight_decay_name}, {"lr_mult": bias_lr_mult,
                                                                                                      "decay_mult": bias_decay_mult}],
                  name = conv_name,
                  kernel_size = fileter_kernel_size,
                  stride = striding,
                  pad = padding,
                  num_output = num_output,
                  group=2,
                  weight_filler=dict(type = weight_filler_type),
                  bias_filler=dict(type = bias_filler_type,value=0))

