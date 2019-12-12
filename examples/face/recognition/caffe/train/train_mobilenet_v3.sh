#!/bin/sh
if ! test -f ../prototxt/mobilenet/train_v3.prototxt ;then
	echo "error: train_v3.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=../prototxt/mobilenet/solver_v3.prototxt -gpu 1 \
#--snapshot=../snapshot/mobilenet_v3_iter_207766.solverstate
