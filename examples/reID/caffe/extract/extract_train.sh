set -e
if [ ! -n "$1" ] ;then
    echo "\$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
if [ ! -n "$2" ] ;then             
    echo "\$2 is empty, default is vgg_reduce"    
    base_model='vgg16'             
else 
    echo "use $2 as base model"    
    base_model=$2   
fi   
     
if [ ! -n "$3" ] ;then             
    echo "\$3 is empty, default is feature_name"  
    feature_name='fc7'             
else 
    echo "use $3 as base model"    
    feature_name=$3 
fi
model_file=./models/market1501/$base_model/snapshot/${base_model}.full_iter_18000.caffemodel

python examples/market1501/extract/extract_feature.py \
	examples/market1501/lists/train.lst \
	examples/market1501/datamat/train.lst.fc7.mat \
	examples/market1501/datamat/train.lst.score.mat \
	--gpu $gpu \
	--model_def ./models/market1501/$base_model/dev.proto \
	--feature_name $feature_name \
	--pretrained_model $model_file \
	--mean_value 97.8286,99.0468,105.606
