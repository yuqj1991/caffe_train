root_dir="/home/resideo/workspace/deepano_face_train"
echo $cur_dir
echo $root_dir
cd $root_dir

redo=1
data_root_dir="/home/resideo/workspace/dataset/facedata"
dataset_name="wider_face"
mapfile="/home/resideo/workspace/deepano_face_train/examples/deepano_face/face_detector/labelmap_face.prototxt"
anno_type="detection"
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
for subset in wider_train wider_val
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/examples/deepano_face/face_detector/scripts/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db $data_root_dir/$dataset_name
done
