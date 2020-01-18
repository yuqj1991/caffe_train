#!/bin/sh
if ! test -f ../prototxt/SSD_300x300/lp_det_train.prototxt ;then
	echo "error: ../prototxt/SSD_300x300/lp_det_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/SSD_300x300/lp_det_test.prototxt ;then
	echo "error: ../prototxt/SSD_300x300300x300/lp_det_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../prototxt/SSD_300x300/solver_det.prototxt -gpu 0 \
#--snapshot=../snapshot/det_iter_110000.solverstate
