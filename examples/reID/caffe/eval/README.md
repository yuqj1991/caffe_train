>>(1)test数据集测评：
>>python generate_pair_file.py --pair_file reid_pairs.txt --result_file reid_list.txt
>>利用reid_similarity.py脚本生成LFW人脸对的余弦相似度得分。  
>>python reid_similarity.py   --image_dir ../../../../../dataset/reId_data/combineData/val --pair_file reid_list.txt  --result_file  reidnet_result.txt  --network deploy.prototxt  --weights deploy.caffemodel  

>>(2)LFW性能分析：  
>>利用roc_curve.py脚本生成结果。  
>> python roc_curve.py gt_reid_list.txt  reidnet_result.txt
