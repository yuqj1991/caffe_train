#!/bin/sh
if ! test -f ../prototxt/train_glass.prototxt ;then
	echo "error: ../prototxt/train_glass.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/test_glass.prototxt ;then
	echo "error: ../prototxt/test_glass.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver_glass.prototxt -gpu 0 \
--snapshot=../snapshot/face_glass_iter_377459.solverstate
