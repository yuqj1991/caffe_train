#!/bin/sh
if ! test -f ../../prototxt/modelfull/face_train_v2.prototxt ;then
	echo "error: ../../prototxt/modelfull/face_train_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../prototxt/modelfull/face_test_v2.prototxt ;then
	echo "error: ../../prototxt/modelfull/face_test_v2.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=../../solver/solver_without_blur_occlu/solver_train_v2.prototxt -gpu 1 \
--snapshot=../../snapshot/face_v2_iter_99569.solverstate
