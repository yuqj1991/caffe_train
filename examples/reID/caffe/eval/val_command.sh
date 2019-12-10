#!/bin/sh
if ! test -f ../../../../../deploy_model/reid.prototxt ;then
	echo "error: ../../../../../deploy_model/reid.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../../../../test_model/reid.prototxt ;then
	echo "error: ../../../../../test_model/reid.prototxt does not exit."
	echo "please generate your own model prototxt primarily."
        exit 1
fi
if ! test -f ../../../../../deploy_model/reid.caffemodel ;then
	echo "error: ../../../../../deploy_model/reid.caffemodel does not exit."
	echo "please generate your own model primarily."
        exit 1
fi
python caffe_to_deploy.py --model ../../../../../deploy_model/reid.prototxt --weight ../../../../../deploy_model/reid.caffemodel --model_name ../../../../../test_model/reid
python reid_similarity.py --image_dir ../../../../../dataset/reId_data/combineData/val --pair_file reid_list.txt  --result_file reidnet_result.txt  --network ../../../../../test_model/reid.prototxt  --weights ../../../../../test_model/reid.caffemodel
python roc_curve.py gt_reid_list.txt  reidnet_result.txt
