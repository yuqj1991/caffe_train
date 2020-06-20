#!/bin/sh
if ! test -f ../prototxt/Full_640x640/Yolov3_face_train.prototxt ;then
	echo "error: ../../prototxt/Full_640x640/Yolov3_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/Full_640x640/Yolov3_face_test.prototxt ;then
	echo "error: ../../prototxt/Full_640x640/Yolov3_face_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/Full_640x640/Yolov3_face_solver.prototxt -gpu 0 \
#--snapshot=../snapshot/face_iter_5000.solverstate
