#!/bin/sh
if ! test -f ../prototxt/deep_ccpd_rec_train_v1.prototxt ;then
	echo "error: ../prototxt/deep_ccpd_rec_train_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deep_ccpd_rec_test_v1.prototxt ;then
	echo "error: ../prototxt/deep_ccpd_rec_test_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../solver/deep_solver_ccpd_rec_v1.prototxt -gpu 1 \
#--snapshot=../snapshot/deepccpd_rec_v1_iter_44952.solverstate
