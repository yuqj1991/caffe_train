#!/bin/sh
if ! test -f ../prototxt/faceattrinet/train_v2.prototxt ;then
	echo "error: ../prototxt/train_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/faceattrinet/test_v2.prototxt ;then
	echo "error: ../prototxt/test_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/faceattri_solver_train_v2.prototxt -gpu 0 \
#--snapshot=../snapshot/facelandmark_v2_iter_185838.solverstate
