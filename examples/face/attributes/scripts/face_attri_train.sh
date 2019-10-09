#!/bin/sh
if ! test -f ../prototxt/faceattrinet/faceattri_train.prototxt ;then
	echo "error: ../prototxt/faceattri_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/faceattrinet/faceattri_test.prototxt ;then
	echo "error: ../prototxt/faceattri_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/faceattri_solver_train.prototxt -gpu 0 \
#--snapshot=../snapshot/facelandmark_v2_iter_185838.solverstate
