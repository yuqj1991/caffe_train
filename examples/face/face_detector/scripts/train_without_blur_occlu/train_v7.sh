#!/bin/sh
if ! test -f ../../prototxt/detect_without_blur_occlu/face_train_v7.prototxt ;then
	echo "error: ../../prototxt/detect_without_blur_occlu/face_train_v7.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../prototxt/detect_without_blur_occlu/face_test_v7.prototxt ;then
	echo "error: ../../prototxt/detect_without_blur_occlu/face_test_v7.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=../../solver/solver_without_blur_occlu/solver_train_v7.prototxt -gpu 0 \
<<<<<<< HEAD
--snapshot=../../snapshot/face_detector_without_v7_iter_210099.solverstate
=======
#--snapshot=../../snapshot/face_detector_without_v7_iter_185098.solverstate
>>>>>>> 12b61bfafe6c7e8df1f8b53ee7a19fa23ecad639
