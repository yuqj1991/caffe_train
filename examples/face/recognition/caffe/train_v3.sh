#!/bin/sh
if ! test -f ./softmax_loss/my_design/face_train_v3.prototxt ;then
	echo "error: train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=./softmax_loss/my_design/solver_v3.prototxt -gpu 2 \
--snapshot=face_recog/tiny_iter_1504.solverstate
