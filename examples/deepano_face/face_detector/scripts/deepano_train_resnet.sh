#!/bin/sh
if ! test -f ../prototxt/deepano_light_face_train_resblock_v2.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_light_face_test_resblock_v2.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/deepano_solver_train_blur_resnet.prototxt -gpu 0,1 \
--snapshot=../snapshot/deepanoFace_res_iter_130301.solverstate
