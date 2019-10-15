import numpy as np
import os
import sys
import argparse
import glob
import time
import _init_paths
from units import SClassifier, AverageMeter, convert_secs2time
import caffe
import scipy.io as sio

def load_txt(xfile):
  img_files = []
  labels = []
  for line in open(xfile):
    line = line.strip('\n').split(' ')
    assert(len(line) == 2)
    img_files.append(line[0])
    labels.append(int(float(line[1])))
  return img_files, labels

def main(argv):

  parser = argparse.ArgumentParser()
  # Required arguments: input and output files.
  parser.add_argument(
    "input_file",
    help="Input image, directory"
  )
  parser.add_argument(
    "feature_file",
    help="Feature mat filename."
  )
  parser.add_argument(
    "score_file",
    help="Score Output mat filename."
  )
  # Optional arguments.
  parser.add_argument(
    "--model_def",
    default=os.path.join(
            "./models/market1501/caffenet/feature.proto"),
    help="Model definition file."
  )
  parser.add_argument(
    "--pretrained_model",
    default=os.path.join(
            "./models/market1501/caffenet/caffenet_iter_17000.caffemodel"),
    help="Trained model weights file."
  )
  parser.add_argument(
    "--gpu",
    type=int,
    default=-1,
    help="Switch for gpu computation."
  )
  parser.add_argument(
    "--center_only",
    action='store_true',
    help="Switch for prediction from center crop alone instead of " +
         "averaging predictions across crops (default)."
  )
  parser.add_argument(
    "--images_dim",
    default='256,256',
    help="Canonical 'height,width' dimensions of input images."
  )
  parser.add_argument(
    "--mean_value",
    default=os.path.join(
                         'examples/market1501/market1501_mean.binaryproto'),
    help="Data set image mean of [Channels x Height x Width] dimensions " +
         "(numpy array). Set to '' for no mean subtraction."
  )
  parser.add_argument(
    "--input_scale",
    type=float,
    help="Multiply input features by this scale to finish preprocessing."
  )
  parser.add_argument(
    "--raw_scale",
    type=float,
    default=255.0,
    help="Multiply raw input by this scale before preprocessing."
  )
  parser.add_argument(
    "--channel_swap",
    default='2,1,0',
    help="Order to permute input channels. The default converts " +
         "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
  )
  parser.add_argument(
    "--ext",
    default='jpg',
    help="Image file extension to take as input when a directory " +
         "is given as the input file."
  )
  parser.add_argument(
    "--feature_name",
    default="fc7",
    help="feature blob name."
  )
  parser.add_argument(
    "--score_name",
    default="prediction",
    help="prediction score blob name."
  )
  args = parser.parse_args()

  image_dims = [int(s) for s in args.images_dim.split(',')]

  channel_swap = None
  if args.channel_swap:
    channel_swap = [int(s) for s in args.channel_swap.split(',')]

  mean_value = None
  if args.mean_value:
    mean_value = [float(s) for s in args.mean_value.split(',')]
    mean_value = np.array(mean_value)

  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    print("GPU mode, device : {}".format(args.gpu))
  else:
    caffe.set_mode_cpu()
    print("CPU mode")

  # Make classifier.
  classifier = SClassifier(args.model_def, args.pretrained_model,
        image_dims=image_dims, mean_value=mean_value,
        input_scale=args.input_scale, raw_scale=args.raw_scale,
        channel_swap=channel_swap)

  # Load numpy, directory glob (*.jpg), or image file.
  args.input_file = os.path.expanduser(args.input_file)
  if args.input_file.endswith(args.ext):
    print("Loading file: %s" % args.input_file)
    inputs = [caffe.io.load_image(args.input_file)]
    labels = [-1]
  elif os.path.isdir(args.input_file):
    print("Loading folder: %s" % args.input_file)
    inputs =[caffe.io.load_image(im_f)
             for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    labels = [-1 for _ in xrange(len(inputs))]
  else:
    ## Image List Files
    print("Loading file: %s" % args.input_file)
    img_files, labels = load_txt(args.input_file)
    inputs = [caffe.io.load_image(im_f)
              for im_f in img_files]

  print("Classifying %d inputs." % len(inputs))

  # Classify.
  ok = 0.0

  save_feature = None
  save_score   = None
  start_time = time.time()
  epoch_time = AverageMeter()
  for idx, _input in enumerate(inputs):
    _ = classifier.predict([_input], not args.center_only)
    feature = classifier.get_blob_data(args.feature_name)
    score   = classifier.get_blob_data(args.score_name)
    assert (feature.shape[0] == 1 and score.shape[0] == 1)
    feature_shape = feature.shape
    score_shape = score.shape
    if save_feature is None:
        print('feature : {} : {}'.format(args.feature_name, feature_shape))
        save_feature = np.zeros((len(inputs), feature.size),dtype=np.float32)
    save_feature[idx, :] = feature.reshape(1, feature.size)
    if save_score is None:
        print('score : {} : {}'.format(args.score_name, score_shape))
        save_score = np.zeros((len(inputs), score.size),dtype=np.float32)
    save_score[idx, :] = score.reshape(1, score.size)

    mx_idx = np.argmax(score.view())
    ok = ok + int(int(mx_idx) == int(labels[idx]))

    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (len(inputs)-idx-1))
    need_time = '{:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs)

    print("{:5d} / {:5d} images, need {:s}. [PRED: {:3d}] vs [OK: {:3d}] accuracy: {:.4f} = good: {:5d} bad: {:5d}".format( \
                                  idx+1, len(inputs), need_time, mx_idx, labels[idx], ok/(idx+1), int(ok), idx+1-int(ok)))
    

  # Save
  if (args.feature_file):
    print("Saving feature into %s" % args.feature_file)
    sio.savemat(args.feature_file, {'feature':save_feature})
  else:
    print("Without saving feature")

  if (args.score_file):
    print("Saving score into %s" % args.score_file)
    sio.savemat(args.score_file, {'feature':save_score})
  else:
    print("Without saving score")


if __name__ == '__main__':
  main(sys.argv)
