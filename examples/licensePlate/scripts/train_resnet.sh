#!/bin/sh
if ! test -f ../prototxt/ResNet/ResNet-train_val.prototxt ;then
	echo "error: ../prototxt/ResNet/ResNet-train_val.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/ResNet/ResNet-train_val.prototxt ;then
	echo "error: ../prototxt/ResNet/ResNet-train_val.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../prototxt/ResNet/solver.prototxt -gpu 0 \
--weights ../net/LPR/lpr_recognition.caffemodel
#--snapshot=../snapshot/resnet_lstm_ctc_iter_18.solverstate
