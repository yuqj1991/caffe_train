#!/bin/sh
if ! test -f ../prototxt/deepano_light_ccpd_rec_train_v1.prototxt ;then
	echo "error: ../prototxt/deepano_light_ccpd_rec_train_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_light_ccpd_rec_test_v1.prototxt ;then
	echo "error: ../prototxt/deepano_light_ccpd_rec_test_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../solver/deepano_solver_ccpd_rec_v1.prototxt -gpu 1 \
--snapshot=../snapshot/deepanoccpd_rec_v1_iter_1564334.solverstate
