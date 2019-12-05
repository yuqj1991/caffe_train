#!/bin/sh
if ! test -f ../prototxt/resnet/train.prototxt ;then
	echo "error: train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=../prototxt/resnet/solver.prototxt -gpu 0 \
#--snapshot=./snapshot/resnet_iter_7479.solverstate
#--weights ./snapshot/resnet_deploy.caffemodel
