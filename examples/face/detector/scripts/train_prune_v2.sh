#!/bin/sh
if ! test -f ../prototxt/model/face_train_v2.prototxt ;then
	echo "error: ../prototxt/model/face_train_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/model/face_test_v2.prototxt ;then
	echo "error: ../prototxt/model/face_test_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver_train_prune_v2.prototxt -gpu 0 \
#--snapshot=../snapshot/face_v2_iter_5000.solverstate
