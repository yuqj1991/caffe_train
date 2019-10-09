#!/bin/sh
if ! test -f ../prototxt/modelPrune/face_train.prototxt ;then
	echo "error: ../prototxt/modelPrune/face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/modelPrune/face_test.prototxt ;then
	echo "error: ../prototxt/modelPrune/face_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver_train.prototxt -gpu 0 \
#--snapshot=../../snapshot/face_detector_without_v8_iter_6791.solverstate
