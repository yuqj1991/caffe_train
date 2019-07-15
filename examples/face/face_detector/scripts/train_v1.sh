#!/bin/sh
if ! test -f ../prototxt/face_train_v1.prototxt ;then
	echo "error: ../prototxt/face_train_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/face_test_v1.prototxt ;then
	echo "error: ../prototxt/face_test_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver_train_v1.prototxt -gpu 0 \
--snapshot=../snapshot/face_detector_v1_iter_15193.solverstate
