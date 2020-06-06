#!/bin/sh
if ! test -f ../prototxt/mobilenet/train_v2.prototxt ;then
	echo "error: train_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=../prototxt/mobilenet/solver_v2.prototxt -gpu 0 \
#--snapshot=../snapshot/mobilenet_v2_iter_812273.solverstate
