#!/bin/sh
if ! test -f ../prototxt/deepano_ccpd_rec_train_v2.prototxt ;then
	echo "error: ../prototxt/deepano_ccpd_rec_train_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_ccpd_rec_test_v2.prototxt ;then
	echo "error: ../prototxt/deepano_ccpd_rec_test_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../solver/deepano_solver_ccpd_rec_v2.prototxt -gpu 0 \
#--snapshot=../snapshot/deepanoccpd_rec_v2_iter_30379.solverstate
