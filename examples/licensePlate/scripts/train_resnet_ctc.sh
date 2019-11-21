#!/bin/sh
if ! test -f ../prototxt/ResNet_LSTM_CTC/lp_rec_train.prototxt ;then
	echo "error: ../prototxt/ResNet_LSTM_CTC/lp_rec_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/ResNet_LSTM_CTC/lp_rec_test.prototxt ;then
	echo "error: ../prototxt/ResNet_LSTM_CTC/lp_rec_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../prototxt/ResNet_LSTM_CTC/solver.prototxt -gpu 0 \
#--snapshot=../snapshot/deepccpd_rec_v1_iter_44952.solverstate
