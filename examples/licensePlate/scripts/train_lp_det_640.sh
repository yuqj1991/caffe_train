#!/bin/sh
if ! test -f ../prototxt/SSD_640x640/MobileNetSSD_train.prototxt ;then
	echo "error: ../prototxt/SSD_640x640/MobileNetSSD_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/SSD_640x640/MobileNetSSD_test.prototxt ;then
	echo "error: ../prototxt/SSD_640x640/MobileNetSSD_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../prototxt/SSD_640x640/solver_train.prototxt -gpu 0 \
#--snapshot=../snapshot/det_iter_110000.solverstate
