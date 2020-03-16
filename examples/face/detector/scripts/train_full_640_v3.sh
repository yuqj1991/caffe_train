#!/bin/sh
if ! test -f ../prototxt/Full_640x640/train_v3.prototxt ;then
	echo "error: ../../prototxt/Full_640x640/train_v3.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/Full_640x640/test_v3.prototxt ;then
	echo "error: ../../prototxt/Full_640x640/test_v3.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/Full_640x640/solver_v3.prototxt -gpu 0 \
#--snapshot=../snapshot/face_iter_5014.solverstate
