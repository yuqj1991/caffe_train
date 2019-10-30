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

class TripletLayer(caffe.Layer):
    
    global no_residual_list,margin
    
    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        
        assert shape(bottom[0].data) == shape(bottom[1].data)
        assert shape(bottom[0].data) == shape(bottom[2].data)

        layer_params = yaml.load(self.param_str_)
        self.margin = layer_params['margin']
        
        self.a = 1
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        anchor_minibatch_db = []
        positive_minibatch_db = []
        negative_minibatch_db = []
        for i in range((bottom[0]).num):
                
            anchor_minibatch_db.append(bottom[0].data[i])
       
            positive_minibatch_db.append(bottom[1].data[i]) 
        
            negative_minibatch_db.append(bottom[2].data[i])

        loss = float(0)
        self.no_residual_list = []
        for i in range(((bottom[0]).num)):
            a = np.array(anchor_minibatch_db[i])
            p = np.array(positive_minibatch_db[i])
            n = np.array(negative_minibatch_db[i])
            a_p = a - p
            a_n = a - n
            ap = np.dot(a_p,a_p)
            an = np.dot(a_n,a_n)
            dist = (self.margin + ap - an)
            _loss = max(dist,0.0)
            if i == 0:
                print ('loss:'+' ap:'+str(ap)+' '+'an:'+str(an))
            if _loss == 0 :
                self.no_residual_list.append(i)
            loss += _loss
        
        loss = (loss/(2*(bottom[0]).num))
        top[0].data[...] = loss
    

    def backward(self, top, propagate_down, bottom):
        count = 0
        if propagate_down[0]:
            for i in range((bottom[0]).num):
                if not i in self.no_residual_list:
                    x_a = bottom[0].data[i]
                    x_p = bottom[1].data[i]
                    x_n = bottom[2].data[i]
                    
                    #print x_a,x_p,x_n
                    bottom[0].diff[i] =  self.a*((x_n - x_p)/((bottom[0]).num))
                    bottom[1].diff[i] =  self.a*((x_p - x_a)/((bottom[0]).num))
                    bottom[2].diff[i] =  self.a*((x_a - x_n)/((bottom[0]).num))
                    
                    count += 1
                else:
                    bottom[0].diff[i] = np.zeros(shape(bottom[0].data)[1])
                    bottom[1].diff[i] = np.zeros(shape(bottom[0].data)[1])
                    bottom[2].diff[i] = np.zeros(shape(bottom[0].data)[1])
        
        #print 'select gradient_loss:',bottom[0].diff[0][0]
        #print shape(bottom[0].diff),shape(bottom[1].diff),shape(bottom[2].diff)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass





