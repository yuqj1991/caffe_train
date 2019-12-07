#!/bin/sh
if ! test -f ../prototxt/ResNet_50_LSTM_CTC/lp_rec_train.prototxt ;then
	echo "error: ../prototxt/ResNet_LSTM_CTC/lp_rec_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/ResNet_50_LSTM_CTC/lp_rec_test.prototxt ;then
	echo "error: ../prototxt/ResNet_LSTM_CTC/lp_rec_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../prototxt/ResNet_50_LSTM_CTC/solver.prototxt -gpu 0 \
#--weights ../snapshot/resnet_lstm_ctc_iter_399649.caffemodel
