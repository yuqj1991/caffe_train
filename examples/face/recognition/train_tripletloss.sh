python3 src/train_tripletloss.py \
--logs_base_dir ./train/logs/ \
--models_base_dir ./train/models/  \
--data_dir ../../../../dataset/facedata/recognition/align_vggface_train/ \
--model_def models.inception_resnet_v1 \
--optimizer RMSPROP \
--image_size 160 \
--batch_size 30 \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
--learning_rate 0.01 \
--weight_decay 1e-4 \
--max_nrof_epochs 50000 \
--epoch_size 10 \
--gpu_memory_fraction 0.7
#--pretrained_model ./train/20180408-102900/
