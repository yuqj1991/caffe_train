#!/bin/sh
if ! test -f ../prototxt/s3fd_mv2/train.prototxt ;then
	echo "error: ../prototxt/s3fd_mv2/train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/s3fd_mv2/test.prototxt ;then
	echo "error: ../prototxt/s3fd_mv2/test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../solver/solver_train_s3fd_mv2.prototxt -gpu 1 \
#--snapshot=../snapshot/face_s3fd_mv2.solverstate
