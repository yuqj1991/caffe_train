#!/bin/sh
if ! test -f ../prototxt/deepano_light_face_train.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_light_face_test.prototxt ;then
	echo "error: ../prototxt/deepano_light_face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/deepano_solver_train.prototxt --snapshot=/home/resideo/workspace/deepano_face_train/examples/deepano_face/face_detector/snapshot/deepanoFace_iter_60000.solverstate -gpu 0
