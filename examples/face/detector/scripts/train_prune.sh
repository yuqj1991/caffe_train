#!/bin/sh
if ! test -f ../prototxt/modelprune/face_train.prototxt ;then
	echo "error: ../prototxt/modelprune/face_train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/modelprune/face_test.prototxt ;then
	echo "error: ../prototxt/modelprune/face_test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver_train_prune.prototxt -gpu 0 \
<<<<<<< HEAD
--snapshot=../snapshot/face_iter_5000.solverstate
=======
#--snapshot=../snapshot/face_iter_105000.solverstate
>>>>>>> 38ee37817b4bbaa94222f18eb150f51cbc3f14b7
