#!/bin/sh
if ! test -f ../prototxt/train_glass_no_pad.prototxt ;then
	echo "error: ../prototxt/train_glass_no_pad.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/test_glass_no_pad.prototxt ;then
	echo "error: ../prototxt/test_glass_no_pad.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver_glass_no_pad.prototxt -gpu 1 \
#--snapshot=../snapshot/face_glass_iter_377459.solverstate
