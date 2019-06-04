#!/bin/sh
if ! test -f ../prototxt/deepano_light_face_train_resblock_v7.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train_resblock_v7.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_light_face_test_resblock_v7.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_test_resblock_v7.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/deepano_solver_train_blur_resnet_v7.prototxt -gpu 3 \
#--snapshot=../snapshot/deepanoFace_res_v6_iter_62734.solverstate
