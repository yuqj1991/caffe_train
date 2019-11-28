>>(1)test数据集测评：  
>>利用reid_similarity.py脚本生成LFW人脸对的余弦相似度得分。  
>>python reid_similarity.py   --image_dir lfw_mtcnnpy_160
>>                            --pair_file reid_list_lfw.txt  
>>                            --result_file  reidnet_vggreid2_inception_resnet_v2_lfw.txt  
>>                            --network inception_resnet_v2_tiny_deploy.prototxt  
>>                            --weights reidnet_inception_resnet_v2_tiny_iter_192000.caffemodel  

>>(2)LFW性能分析：  
>>利用roc_curve.py脚本生成结果。  
>> python roc_curve.py gt_reid_list_lfw.txt  reidnet_vggreid2_inception_resnet_v2_lfw.txt
