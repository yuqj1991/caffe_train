from __future__ import print_function
import argparse
import sys
caffe_root = '../../../caffe_train/'
sys.path.insert(0, caffe_root + 'python') 
import caffe
import numpy as np

from datasets.factory import get_imdb
from utils.timer import Timer
from utils.get_config import get_output_dir
import cv2

def parser():
    parser = argparse.ArgumentParser('face Evaluate Module!',
                            description='You can use this file to evaluate face model capbilitaty!')
    parser.add_argument('--db', dest='db_name', help='Path to the image',
                        default='wider_val', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='The GPU ide to be used',
                        default=0, type=int)
    parser.add_argument('--prototxt', dest='prototxt', help='face caffe test prototxt',
                        default='net/face_detector.prototxt', type=str)
    parser.add_argument('--out_path', dest='out_path', help='Output path for saving the figure',
                        default='output', type=str)
    parser.add_argument('--model', dest='model', help='face trained caffemodel',
                        default='net/face_detector.caffemodel', type=str)
    parser.add_argument('--net_name', dest='net_name',
                        help='The name of the experiment',
                        default='face_detector',type=str)
    return parser.parse_args()


def preprocess(src):
    img = cv2.resize(src, (320,320))
    img = img - [103.94, 116.78, 123.68] # 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)


def detect(net, im_path, thresh=0.05, timers=None):
    """
    Main module to detect faces
    :param net: The trained network
    :param im_path: The path to the image
    :param thresh: Detection with a less score than thresh are ignored
    :param timers: Timers for calculating detect time (if None new timers would be created)
    :return: dets (bounding boxes concatenated with scores) and the timers
    """
    if not timers:
        timers = {'detect': Timer(),
                  'misc': Timer()}

    im = cv2.imread(im_path)
    sys.stdout.flush()
    timers['detect'].tic()

    img = preprocess(im)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    out = net.forward()
    boxes, conf, cls = postprocess(im, out)
    timers['detect'].toc()
    timers['misc'].tic()
    inds = np.where(conf[:] > thresh)[0]
    probs = conf[inds]
    boxes = boxes[inds, :]
    dets = np.hstack((boxes, probs[:, np.newaxis])).astype(np.float32, copy=False)
    timers['misc'].toc()
    return dets,timers


def test_net(net, imdb, thresh=0.05, output_path=None):
    """
    Testing the SSH network on a dataset
    :param net: The trained network
    :param imdb: The test imdb
    :param thresh: Detections with a probability less than this threshold are ignored
    :param output_path: Output directory
    """
    # Initializing the timers
    print('Evaluating {} on {}'.format(net.name,imdb.name))
    timers = {'detect': Timer(), 'misc': Timer()}
    run_inference = True
    dets = [[[] for _ in xrange(len(imdb))] for _ in xrange(imdb.num_classes)]
    output_dir = get_output_dir(imdb_name=imdb.name, net_name=net.name,output_dir=output_path)
    print('output: ', output_dir)
    # Perform inference on images if necessary
    if run_inference:
        for i in xrange(len(imdb)):
            im_path =imdb.image_path_at(i)
            dets[1][i], detect_time = detect(net, im_path, thresh, timers=timers)
            print('\r{:d}/{:d} detect-time: {:.3f}s, misc-time:{:.3f}s'.format(i + 1, len(imdb), timers['detect'].average_time,timers['misc'].average_time), end='')
        print('\n', end='')

    # Evaluate the detections
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes=dets, output_dir=output_dir, method_name=net.name)
    print('All Done!')


def main(args):
    # Loading the network
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)

    # Create the imdb
    imdb = get_imdb(args.db_name)

    # Set the network name
    net.name = args.net_name

    # Evaluate the network
    test_net(net, imdb, output_path=args.out_path)


if __name__ == '__main__':
    args = parser()
    main(args)
