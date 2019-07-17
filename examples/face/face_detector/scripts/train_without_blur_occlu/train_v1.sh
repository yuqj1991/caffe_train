#!/bin/sh
if ! test -f ../../prototxt/detect_without_blur_occlu/face_train_v1.prototxt ;then
	echo "error: ../../prototxt/detect_without_blur_occlu/face_train_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../prototxt/detect_without_blur_occlu/face_test_v1.prototxt ;then
	echo "error: ../../prototxt/detect_without_blur_occlu/face_test_v1.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=../../solver/solver_without_blur_occlu/solver_train_v1.prototxt -gpu 0 \
#--snapshot=../../snapshot/face_detector_without_v1_iter_6791.solverstate
