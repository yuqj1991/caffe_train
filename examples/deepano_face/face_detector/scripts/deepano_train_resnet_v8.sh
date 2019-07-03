#!/bin/sh
if ! test -f ../prototxt/deepano_face_train_v8.prototxt ;then
	echo "error: ../prototxt/deepano_face_train_v8.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/deepano_face_test_v8.prototxt ;then
	echo "error: ../prototxt/deepano_face_test_v8.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
<<<<<<< HEAD
../../../../build/tools/caffe train --solver=../solver/deepano_solver_train_v8.prototxt -gpu 2 \
=======
../../../../build/tools/caffe train --solver=../solver/deepano_solver_train_v8.prototxt -gpu 0 \
>>>>>>> d8059256142d67f3ac6f1e8b6c388145443537fd
#--snapshot=../snapshot/deepanoface_v8_iter_3083.solverstate
