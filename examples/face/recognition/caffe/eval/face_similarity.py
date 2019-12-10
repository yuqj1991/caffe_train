# -*- coding: utf-8 -*
import numpy as np
import sys
import os
import cv2
import argparse

import logging

try:
    caffe_root = '../../../../../../caffe_train/'
    sys.path.insert(0, caffe_root + 'python')
    import caffe
except ImportError:
    logging.fatal("Cannot find caffe!")
import time



def l2_normalize(vector):
    output = vector/np.sqrt(max(np.sum(vector**2), 1e-12))
    return output

def cos(vector1,vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)

def caffenet_load(net_file, weights_file, mode):
    if mode == 'GPU':
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(net_file, weights_file, caffe.TEST)

    return net

def image_pair_build(image_path, pair_path):

    image_pair_path = []
    #image_pair_label = []

    with open(pair_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            path1, path2 = line.strip('\n').split(" ")
            image_pair_path += (os.path.join(image_path, path1), os.path.join(image_path, path2))
            print((os.path.join(image_path, path1), os.path.join(image_path, path2)))

    return image_pair_path


def preprocess(img, inputSize):

    preprocessed_image = cv2.resize(img, inputSize)
    preprocessed_image = np.transpose(preprocessed_image, (2,0,1))
    preprocessed_image = preprocessed_image.astype("float")
    preprocessed_image = preprocessed_image - 127.5
    preprocessed_image = preprocessed_image * 0.0078125

    return preprocessed_image


def face_similarity_result(similarity_list, similairty_file):

    print("similarity_list = {}".format(len(similarity_list)))

    with open(similairty_file, "w") as f:
        for i in range(0, len(similarity_list)):
            f.write("{}\n".format(similarity_list[i]))


def main(args):
    if not os.path.exists(args.image_dir):
        print("{} does not exist".format(args.image_dir))
        exit()

    if not os.path.exists(args.pair_file):
        print("{} does not exist".format(args.pair_file))
        exit()

    if not os.path.exists(args.network):
        print("{} does not exist".format(args.network))
        exit()

    if not os.path.exists(args.weights):
        print("{} does not exist".format(args.weights))
        exit()

    facenet = caffenet_load(args.network, args.weights, "GPU")
    image_paths = image_pair_build(args.image_dir, args.pair_file)
    inputShape = facenet.blobs['data'].data.shape
    inputSize = (inputShape[3], inputShape[2])

    total_pair = len(image_paths)//2
    print("total_pair: ", total_pair)
    similarity_list = []
    for idx in range(total_pair):
        print("image_left: %s, image_right: %s"%(image_paths[2*idx], image_paths[2*idx+1]))
        image_left = preprocess(cv2.imread(image_paths[2*idx]), inputSize)
        image_right = preprocess(cv2.imread(image_paths[2*idx + 1]), inputSize)
	
        image_left = image_left[np.newaxis, :]
        image_right = image_right[np.newaxis, :]
        images = np.concatenate((image_left, image_right))

        facenet.blobs['data'].data[...] = images
        embeddings = facenet.forward()['fc5']
        embedding_left = embeddings[0]
        embedding_right = embeddings[1]
        # print(embedding_left)
        # print(embedding_right)
        # exit()


        norm_left = l2_normalize(embedding_left)
        norm_right = l2_normalize(embedding_right)

        cosine = cos(norm_left, norm_right)
        if cosine < 0.0:
            cosine =0.0
        print(cosine)
        similarity_list.append(cosine)

    face_similarity_result(similarity_list, args.result_file)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Directory with face aligned images.')
    parser.add_argument('--pair_file', type=str, help='File of face pairs list')
    parser.add_argument('--result_file', type=str, help='Result of face recognition')
    parser.add_argument('--network', type=str, help='Network file of face recognition')
    parser.add_argument('--weights', type=str, help='Weights file of face recognition')

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
