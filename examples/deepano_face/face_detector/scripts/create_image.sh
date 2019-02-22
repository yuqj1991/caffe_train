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
for subset in wider_test
do
  $root_dir/build/tools/convert_imageset $data_root_dir $root_dir/examples/deepano_face/face_detector/scripts/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db
done
