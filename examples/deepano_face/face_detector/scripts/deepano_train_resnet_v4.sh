#!/bin/sh
if ! test -f ../prototxt/deepano_face_train_v4.prototxt ;then
	echo "error: ../prototxt/deepano_face_train_v4.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_face_test_v4.prototxt ;then
	echo "error: ../prototxt/deepano_face_test_v4.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/deepano_solver_train_v4.prototxt -gpu 0 \
#--snapshot=../snapshot/face_detector_v4_iter_19055.solverstate
