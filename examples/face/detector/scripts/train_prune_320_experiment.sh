#!/bin/sh
if ! test -f ../prototxt/Prune_320x320/train_v2_experiment.prototxt ;then
	echo "error: ../prototxt/Prune_320x320/train_v2_experiment.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../prototxt/Prune_320x320/test_v2_experiment.prototxt ;then
	echo "error: ../prototxt/Prune_320x320/test_v2_experiment.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/Prune_320x320/solver_v2_experiment.prototxt -gpu 0 \
#--snapshot=../snapshot/face_v6_iter_15924.solverstate
