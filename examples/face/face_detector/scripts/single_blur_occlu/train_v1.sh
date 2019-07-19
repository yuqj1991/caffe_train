#!/bin/sh
if ! test -f ../../prototxt/single_blur_occlu/train_crop.prototxt ;then
	echo "error: ../../prototxt/single_blur_occlu/train_crop.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../prototxt/single_blur_occlu/test_crop.prototxt ;then
	echo "error: ../../prototxt/single_blur_occlu/test_crop.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../../build/tools/caffe train --solver=../../solver/solver_blur_occlu/solver_train_v1.prototxt -gpu 2 \
--snapshot=../../snapshot/single_blur_occlu_iter_55000.solverstate
