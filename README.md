# Triplet loss for Caffe
# cosface loss for caffe
# arc face loss for caffe
Introduce triplet loss layer to caffe.<br>
Concretely, we use cosine matric to constrain the distance between samples among same label/different labels.
# 增加 lstm版本
# lstm使用的是junhyukoh实现的lstm版本（lstm_layer_Junhyuk.cpp/cu），原版不支持变长输入的识别。输入的shape由(TxN)xH改为TxNxH以适应ctc的输入结构。<br>
# WarpCTCLossLayer去掉了对sequence indicators依赖（训练时CNN输出的结构是固定的），简化了网络结构（不需要sequence indicator layer）。<br>
# densenet修改了对Reshape没有正确响应的bug，实现了对变长输入预测的支持。<br>
# 增加transpose_layer、reverse_layer，实现对CNN feature map与lstm输入shape的适配<br>
