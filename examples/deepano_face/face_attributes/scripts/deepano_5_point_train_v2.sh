#!/bin/sh
if ! test -f ../prototxt/facelandmarknet/facelandmark_train_v2.prototxt ;then
	echo "error: ../prototxt/facelandmark_train_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/facelandmarknet/facelandmark_test_v2.prototxt ;then
	echo "error: ../prototxt/facelandmark_test_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/facelandmark_solver_train_v2.prototxt -gpu 0 \
#--snapshot=../snapshot/deepanoFacelandmark_v2_iter_1000808.solverstate
