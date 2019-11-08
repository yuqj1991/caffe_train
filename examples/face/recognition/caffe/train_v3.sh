#!/bin/sh
if ! test -f ./softmax_loss/mobilenet_v2/train.prototxt ;then
	echo "error: train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=./softmax_loss/mobilenet_v2/solver.prototxt -gpu 2 \
#--snapshot=./face_recog/tiny_iter_148441.solverstate
