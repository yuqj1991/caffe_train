#!/bin/sh
if ! test -f train.prototxt ;then
	echo "error: train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=solver.prototxt -gpu 2 \
--snapshot=face_recog/tiny_iter_9740.solverstate
