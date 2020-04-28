#!/bin/sh
if ! test -f ../prototxt/Full_640x640/CenterGridSoftmax_face_train_v2.prototxt ;then
	echo "error: ../prototxt/Full_640x640/CenterGrid_face_train_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/Full_640x640/CenterGridSoftmax_face_test_v2.prototxt ;then
	echo "error: ../prototxt/Full_640x640/CenterGrid_face_test_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/Full_640x640/CenterGridSoftmax_face_solver_v2.prototxt -gpu 0 \
#--snapshot=../snapshot/CenterGridSoftmax_face_iter_5000.solverstate
