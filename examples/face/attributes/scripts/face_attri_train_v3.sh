#!/bin/sh
if ! test -f ../prototxt/train_v3.prototxt ;then
	echo "error: ../prototxt/train_v3.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/test_v3.prototxt ;then
	echo "error: ../prototxt/test_v3.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/faceattri_solver_train_v3.prototxt -gpu 1 \
#--snapshot=../snapshot/faceattri_v2_iter_27715.solverstate
