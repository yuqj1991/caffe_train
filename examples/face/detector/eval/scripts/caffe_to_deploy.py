import os
import sys
import argparse
import logging

try:
    caffe_root = '../../../../../caffe_train/'
    sys.path.insert(0, caffe_root + 'python')
    import caffe
except ImportError:
    logging.fatal("Cannot find caffe!")

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file for inference')
    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    net = caffe.Net(args.model, args.weights, caffe.TEST)
    net.save("no_loss.caffemodel")
