#!/bin/sh 
if ! test -f ../prototxt/Full_320x320/CenterGridSoftmax_face_v3_train.prototxt ;then 
   echo "error: ../prototxt/Full_320x320/CenterGridSoftmax_face_v3_train.prototxt does not exit." 
   echo "please generate your own model prototxt primarily." 
   exit 1
fi
if ! test -f ../prototxt/Full_320x320/CenterGridSoftmax_face_v3_test.prototxt ;then 
   echo "error: ../prototxt/Full_320x320/CenterGridSoftmax_face_v3_test.prototxt does not exit." 
   echo "please generate your own model prototxt primarily." 
   exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/Full_320x320/CenterGridSoftmax_face_v3_solver.prototxt --gpu 1 \
# --snapshot=../snapshot/CenterGridSoftmax_face_v3_0_iter_5000.solverstate 