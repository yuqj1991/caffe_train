#!/bin/sh
if ! test -f ../prototxt/faceposenet/faceposenet_res_inception_train.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/faceposenet/faceposenet_res_inception_test.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/facepose_solver_train.prototxt -gpu 0 \
#--snapshot=../snapshot/deepanoFacelandmark_iter_352891.solverstate
