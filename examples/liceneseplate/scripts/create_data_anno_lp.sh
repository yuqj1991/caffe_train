root_dir="../../../../caffe_train"
cd $root_dir/scripts

redo=1
data_root_dir="../../dataset/car_person_data/car_license"
dataset_name="ccpd_dataset"
mapfile="../examples/deep_liceneseplate/scripts/labelmap_lp.prototxt"
anno_type="Rec_ccpd"
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
for subset in training_lp testing_lp
do
  python create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --label-type=$label_type --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir ../examples/deep_liceneseplate/scripts/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db $data_root_dir/$dataset_name
done
