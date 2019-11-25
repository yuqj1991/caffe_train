#!/bin/sh
if ! test -f ../prototxt/Prune_320x320/train_v1.prototxt ;then
	echo "error: ../prototxt/Prune_320x320/train_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/Prune_320x320/test_v1.prototxt ;then
	echo "error: ../prototxt/Prune_320x320/face_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/Prune_320x320/solver_v1.prototxt -gpu 0 \
#--weights ../snapshot/face_iter_80000.caffemodel
#--snapshot=../snapshot/face_iter_100000.solverstate

