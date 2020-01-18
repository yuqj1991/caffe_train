#!/bin/sh
if ! test -f ../prototxt/train.prototxt ;then
	echo "error: ../prototxt/train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/test.prototxt ;then
	echo "error: ../prototxt/test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver.prototxt -gpu 0 \
#--snapshot=../snapshot/faceattri_iter_69754.solverstate
