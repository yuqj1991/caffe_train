# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""


import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing


class Norm2Layer(caffe.Layer):
    """norm2 layer used for L2 normalization."""

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        
        top[0].reshape(bottom[0].num, shape(bottom[0].data)[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        minibatch_db = []
        
        for i in range((bottom[0]).num):
            X_normalized = preprocessing.normalize(bottom[0].data[i].reshape(1,-1), norm='l2')[0]
            minibatch_db.append(X_normalized)
        #print 'bottom**:',np.dot(bottom[0].data[0],bottom[0].data[0])
        top[0].data[...] = minibatch_db

    def backward(self, top, propagate_down, bottom):
        """This layer does not need to backward propogate gradient"""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
