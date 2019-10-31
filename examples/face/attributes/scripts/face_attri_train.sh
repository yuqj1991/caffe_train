#!/bin/sh
if ! test -f ../prototxt/faceattrinet/train.prototxt ;then
	echo "error: ../prototxt/train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/faceattrinet/test.prototxt ;then
	echo "error: ../prototxt/test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/faceattri_solver_train.prototxt -gpu 0 \
<<<<<<< HEAD
#--snapshot=../snapshot/faceattri_iter_47290.solverstate
=======
#--snapshot=../snapshot/faceattri_iter_158573.solverstate
>>>>>>> 7b089c81027a0358284eabaf080babab0c3696d3
