#!/bin/sh
if ! test -f ../prototxt/S3FD/train.prototxt ;then
	echo "error: ../prototxt/S3FD/train.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/S3FD/test.prototxt ;then
	echo "error: ../prototxt/S3FD/test.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/S3FD/solver.prototxt -gpu 0 \
#--snapshot=../snapshot/face_s3fd_mv2_iter_140671.solverstate
