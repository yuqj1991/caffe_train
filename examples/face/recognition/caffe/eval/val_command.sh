#!/bin/sh
if ! test -f ../../../../../../deploy_model/facenet.prototxt ;then
	echo "error: ../../../../../../deploy_model/facenet.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../../../../../test_model/facenet.prototxt ;then
	echo "error: ../../../../../../test_model/facenet.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../../../../../deploy_model/facenet.caffemodel ;then
	echo "error: ../../../../../../deploy_model/facenet.caffemodel does not exit."
	echo "please generate your own model primarily."
        exit 1
fi
python caffe_to_deploy.py --model ../../../../../../deploy_model/facenet.prototxt --weight ../../../../../../deploy_model/facenet.caffemodel --model_name ../../../../../../test_model/facenet
python face_similarity.py --image_dir ../../../../../../dataset/facedata/lfw/lfw_align_160/ --pair_file face_list_lfw.txt  --result_file face_result.txt  --network ../../../../../../test_model/facenet.prototxt  --weights ../../../../../../test_model/facenet.caffemodel
python roc_curve.py gt_face_list_lfw.txt  face_result.txt
