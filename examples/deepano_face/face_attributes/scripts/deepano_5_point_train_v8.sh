#!/bin/sh
if ! test -f ../prototxt/facelandmarknet/facelandmark_train_v8.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/facelandmarknet/facelandmark_test_v8.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/facelandmark_solver_train_v8.prototxt -gpu 3 \
--snapshot=../snapshot/deepanoFacelandmark_v8_iter_681912.solverstate
