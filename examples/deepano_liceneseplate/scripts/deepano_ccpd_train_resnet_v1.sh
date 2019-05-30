#!/bin/sh
if ! test -f ../prototxt/deepano_light_ccpd_train_resblock_v1.prototxt ;then
	echo "error: ../prototxt/deepano_light_ccpd_train_resblock_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_light_ccpd_test_resblock_v1.prototxt ;then
	echo "error: ../prototxt/deepano_light_ccpd_test_resblock_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../build/tools/caffe train --solver=../solver/deepano_solver_train_blur_resnet_v1.prototxt -gpu 2 \
--snapshot=../snapshot/deepanoccpd_res_v1_iter_35207.solverstate
