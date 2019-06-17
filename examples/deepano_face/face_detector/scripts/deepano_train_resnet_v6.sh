#!/bin/sh
if ! test -f ../prototxt/deepano_face_train_v6.prototxt ;then
	echo "error: ../prototxt/deepano_face_train_v6.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_face_test_v6.prototxt ;then
	echo "error: ../prototxt/deepano_face_test_v6.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/deepano_solver_train_v6.prototxt -gpu 0 \
--snapshot=../snapshot/deepanoface_v6_iter_360000.solverstate
