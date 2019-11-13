#!/bin/sh
if ! test -f ../prototxt/mobilenet_v2/face_train_v3.prototxt ;then
	echo "error: ../prototxt/mobilenet_v2/face_train_v3.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/mobilenet_v2/face_test_v3.prototxt ;then
	echo "error: ../prototxt/mobilenet_v2/face_test_v3.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver_train_mobilenet_v2.prototxt -gpu 1 \
#--snapshot=../snapshot/face_v3_iter_44807.solverstate
