#!/bin/sh
if ! test -f ../prototxt/facelandmarknet/facelandmark_train_v1.prototxt ;then
	echo "error: ../prototxt/facelandmark_train_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/facelandmarknet/facelandmark_test_v1.prototxt ;then
	echo "error: ../prototxt/facelandmark_test_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/facelandmark_solver_train_v1.prototxt -gpu 0 \
--snapshot=../snapshot/facelandmark_v1_iter_14315.solverstate
