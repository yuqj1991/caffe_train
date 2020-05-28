#!/bin/sh 
if ! test -f ../prototxt/Full_640x640/CenterGridSoftmax_face_v2_train.prototxt ;then 
   echo "error: ../prototxt/Full_640x640/CenterGridSoftmax_face_v2_train.prototxt does not exit." 
   echo "please generate your own model prototxt primarily." 
   exit 1
fi
if ! test -f ../prototxt/Full_640x640/CenterGridSoftmax_face_v2_test.prototxt ;then 
   echo "error: ../prototxt/Full_640x640/CenterGridSoftmax_face_v2_test.prototxt does not exit." 
   echo "please generate your own model prototxt primarily." 
   exit 1
fi
../../../../build/tools/caffe train --solver=../prototxt/Full_640x640/CenterGridSoftmax_face_v2_solver.prototxt --gpu 1 \
# --snapshot=../snapshot/CenterGridSoftmax_face_v2_0_iter_5000.solverstate 