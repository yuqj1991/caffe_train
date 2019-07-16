root_dir="../../../../../caffe_deeplearning_train"
cd $root_dir/scripts

redo=1
data_root_dir="../../dataset/facedata"
dataset_name="mtfl"
mapfile="../examples/face/face_attributes/labelmap_face.prototxt"
anno_type="faceattributes"
label_type="txt"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in training testing
do
  python create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --label-type=$label_type --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir ../examples/face/face_attributes/scripts/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db $data_root_dir/$dataset_name
done
