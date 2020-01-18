#!/bin/sh
if ! test -f ../prototxt/SSD_320x320/train.prototxt ;then
	echo "error: ../prototxt/SSD_320x320/train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/SSD_320x320/test.prototxt ;then
	echo "error: ../prototxt/SSD_320x320/test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/SSD_320x320/solver.prototxt -gpu 0 \
#--snapshot=../snapshot/face_v6_iter_15924.solverstate
