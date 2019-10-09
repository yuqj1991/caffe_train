#!/bin/sh
if ! test -f ../../prototxt/detect_without_blur_occlu/face_train_v8.prototxt ;then
	echo "error: ../../prototxt/detect_without_blur_occlu/face_train_v8.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../prototxt/detect_without_blur_occlu/face_test_v8.prototxt ;then
	echo "error: ../../prototxt/detect_without_blur_occlu/face_test_v8.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=../../solver/solver_without_blur_occlu/solver_train_v8.prototxt -gpu 1 \
#--snapshot=../../snapshot/face_detector_without_v8_iter_6791.solverstate
