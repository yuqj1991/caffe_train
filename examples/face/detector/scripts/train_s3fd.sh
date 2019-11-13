#!/bin/sh
if ! test -f ../prototxt/s3fd/train.prototxt ;then
	echo "error: ../prototxt/s3fd_mv2/train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/s3fd/test.prototxt ;then
	echo "error: ../prototxt/s3fd_mv2/test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/s3fd/solver_train_s3fd.prototxt -gpu 0 \
#--snapshot=../snapshot/face_s3fd_mv2.solverstate
