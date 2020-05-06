import os,sys,logging
try:
    caffe_root = '/home/stive/workspace/caffe_train/'
    sys.path.insert(0, caffe_root + 'python')
    import caffe
except ImportError:
    logging.fatal("Cannot find caffe!")
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from caffe.utils import *
import math


def SEMoudleBlock(net, from_layer, channels, layerPrefix= '', ratio = 0.2):
    assert from_layer in net.keys()
    Global_poolingName= "{}_GloabalPool".format(layerPrefix)
    net[Global_poolingName] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
    Full_inproductName_Project = "{}_FullinProduct_Project".format(layerPrefix)
    net[Full_inproductName_Project] = L.InnerProduct(net[Global_poolingName], num_output = channels * ratio)
    Relu_Name = "{}_Relu".format(layerPrefix)
    net[Relu_Name] = L.ReLU6(net[Full_inproductName_Project], in_place=True)
    Full_inproductName_expand = "{}_FullinProduct_Expand".format(layerPrefix)
    net[Full_inproductName_expand] = L.InnerProduct(net[Relu_Name], num_output = channels)
    Sigmoid_Name = "{}_Sigmoid".format(layerPrefix)
    net[Sigmoid_Name] = L.Sigmoid(net[Full_inproductName_expand], in_place= True)
    Scale_name = "{}_AttentionScale".format(layerPrefix)
    net[Scale_name] = L.AttentionScale(net[from_layer], net[Sigmoid_Name])
    return Scale_name


def BiFPNBlock(net, from_layers= [], image_size = 640, min_level = 3, max_level = 7, fpn_cell_repeats = 3, 
                    fpn_out_channels = 88, use_global_stats = True, use_relu = False, 
                    apply_bn=True, is_training=True, conv_after_downsample=False, 
                    use_nearest_resize=False, pooling_type= None):
    assert from_layers in net.keys()
    feat_sizes = get_feat_sizes(image_size, max_level)
    num_levels = max_level - min_level + 1
    feats = []
    # get FPN Layer
    for i in range(num_levels):
        if i < len(from_layers):
            feats.append(from_layers[i])
        else:
            feats.append(resample_feature_map(net, feats[-1], use_global_stats= use_global_stats, 
                              use_relu= False, 
                              target_height= (net.blobs[feats[-1]].shape[2] - 1) // 2 + 1, 
                              target_width= (net.blobs[feats[-1]].shape[3]  - 1) // 2 + 1,
                              target_channels= fpn_out_channels, 
                              current_height= net.blobs[feats[-1]].shape[2], 
                              current_width= net.blobs[feats[-1]].shape[3],  
                              current_channels = net.blobs[feats[-1]].shape[1], layerPrefix = '', 
                              apply_bn= apply_bn, is_training= is_training, conv_after_downsample=conv_after_downsample, 
                              use_nearest_resize= use_nearest_resize, pooling_type= pooling_type)                  
    _verify_feats_size(net= net, feats= feats, feat_sizes= feat_sizes, min_level= min_level, max_level= max_level)
    # need Upsampler node
    pnodes = [
        {'feat_level': 6, 'inputs_offsets': [3, 4]},
        {'feat_level': 5, 'inputs_offsets': [2, 5]},
        {'feat_level': 4, 'inputs_offsets': [1, 6]},
        {'feat_level': 3, 'inputs_offsets': [0, 7]},
        {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},
        {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},
        {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},
        {'feat_level': 7, 'inputs_offsets': [4, 11]},
    ]
    for repeate_idx in range(fpn_cell_repeats):
        new_feats = BuildBiFPNLayer(net= net, feats= feats, feat_sizes= feat_sizes, fpn_nodes= pnodes, layerPrefix = ,
                                        fpn_out_filters= fpn_out_channels, min_level = min_level, max_level = max_level, 
                                        use_global_stats = use_global_stats, use_relu = use_relu, concat_method= "fast_attention", 
                                        apply_bn=apply_bn, is_training=is_training, conv_after_downsample=conv_after_downsample,
                                        separable_conv = True, use_nearest_resize=use_nearest_resize, pooling_type= pooling_type)

        feats = [new_feats[level] for level in range(min_level, max_level + 1)]
        _verify_feats_size(net= net, feats= feats, feat_sizes= feat_sizes, min_level= min_level, max_level= max_level)
    return new_feats
        


def MBottleConvBlock(net, from_layer, id, repeated_num, fileter_channels, strides, expansion_factor, kernel_size= 3,
                        Use_BN = True, Use_scale = True, use_global_stats= False, Use_SE= False, **bn_param):
    if strides == 1:
        out_layer_expand = "conv_{}_{}/{}".format(id, repeated_num, "expand")
        ConvBNLayer(net, from_layer, out_layer_expand, use_bn=Use_BN, use_relu = True,
                    num_output = fileter_channels * expansion_factor, kernel_size=1, 
                    pad=0, stride = strides, use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        out_layer_depthswise = "conv_{}_{}/{}".format(id, repeated_num, "depthwise")
        ConvBNLayer(net, out_layer_expand, out_layer_depthswise, use_bn=Use_BN, use_relu=True,
                    num_output = fileter_channels * expansion_factor, kernel_size=kernel_size, pad=1, 
                    group= fileter_channels * expansion_factor,
                    stride = strides, use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        out_layer = out_layer_depthswise
        if Use_SE:
            out_layer = SEMoudleBlock(net, from_layer= out_layer, channels= fileter_channels * expansion_factor, 
                                        layerPrefix= 'SE_{}_{}/{}'.format(id, repeated_num, 'attention'), ratio = 0.25) 
        out_layer_projects = "conv_{}_{}/{}".format(id, repeated_num, "linear")
        ConvBNLayer(net, out_layer, out_layer_projects, use_bn=Use_BN, use_relu=False,
                    num_output = fileter_channels, kernel_size=1, pad=0, stride = strides, 
                    use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        res_name = 'Res_Sum_{}_{}'.format(id, repeated_num)
        net[res_name] = L.Eltwise(net[from_layer], net[out_layer_projects])
        return res_name
    elif strides == 2:
        out_layer_expand = "conv_{}_{}/{}".format(id, repeated_num, "expand")
        ConvBNLayer(net, from_layer, out_layer_expand, use_bn=Use_BN, use_relu=True,
                    num_output = fileter_channels * expansion_factor, kernel_size=1, pad=0, stride = 1, 
                    use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        out_layer_depthswise = "conv_{}_{}/{}".format(id, repeated_num, "depthwise")
        ConvBNLayer(net, out_layer_expand, out_layer_depthswise, use_bn=Use_BN, use_relu=True,
                    num_output = fileter_channels * expansion_factor, kernel_size=kernel_size, pad=0, stride = strides, 
                    group= fileter_channels * expansion_factor, use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        out_layer_projects = "conv_{}_{}/{}".format(id, repeated_num, "linear")
        ConvBNLayer(net, out_layer_depthswise, out_layer_projects, use_bn=Use_BN, use_relu=False,
                    num_output = fileter_channels, kernel_size=1, pad=0, stride = 1, use_scale = Use_scale
                    , use_global_stats= use_global_stats,
                    **bn_param)
        return out_layer_projects


def MobilenetV2Body(net, from_layer, Use_BN = True, **bn_param):
    assert from_layer in net.keys()
    index = 0
    out_layer = "conv_{}".format(index)
    ConvBNLayer(net, from_layer, out_layer, use_bn=Use_BN, use_relu=True,
                num_output= 32, kernel_size=3, pad=1, stride = 2, use_scale = True,
                **bn_param)
    index += 1
    ################################
    # t c  n s
    # - 32 1 2
    # 1 16 1 1
    # 6 24 2 2
    # 6 32 3 2
    # 6 64 4 2
    # 6 96 3 1
    # 6 160 3 2
    # 6 320 1 1
    ###############################
    Inverted_residual_setting = [[1, 16, 1, 1],
                                 [6, 24, 2, 2],
                                 [6, 32, 3, 2],
                                 [6, 64, 4, 2],
                                 [6, 96, 3, 1],
                                 [6, 160, 3, 2],
                                 [6, 320, 1, 1]]
    for _, (t, c, n, s) in enumerate(Inverted_residual_setting):
        if n > 1:
            if s == 2:
                layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, Use_BN = True, Use_scale = True, **bn_param)
                out_layer = layer_name
                index += 1
                strides = 1
                for id in range(n - 1):
                    layer_name = MBottleConvBlock(net, out_layer, index, id, c, strides, t, Use_BN = True, Use_scale = True, **bn_param)
                    out_layer = layer_name
                    index += 1
            elif s == 1:
                for id in range(n):
                    layer_name = MBottleConvBlock(net, out_layer, index, id, c, s, t, Use_BN = True, Use_scale = True, **bn_param)
                    out_layer = layer_name
                    index += 1
        elif n == 1:
            assert s == 1
            layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, Use_BN = True, Use_scale = True, **bn_param)
            out_layer = layer_name
            index += 1
    return net


def ResConnectBlock(net, from_layer_one, from_layer_two, stage_idx, use_global_stats):
    res_name = "ResConnect_stage_{}".format(stage_idx)
    net[res_name] = L.Eltwise(net[from_layer_one], net[from_layer_two], operation = P.Eltwise.SUM)
    out_layer = "Dectction_stage_{}".format(stage_idx)
    ConvBNLayer(net, res_name, out_layer, use_bn = False, use_relu = False, 
                num_output= 6, kernel_size=1, pad = 0, 
                stride=1, use_scale = False, lr_mult=1, use_global_stats= use_global_stats)
    return res_name, out_layer


def CenterGridObjectLoss(net, bias_scale, low_bbox_scale, up_bbox_scale, 
                         stageidx, from_layers = [], net_height = 640, net_width = 640,
                         normalization_mode = P.Loss.VALID, num_classes= 2, loc_weight = 1.0, 
                         share_location = True, ignore_thresh = 0.3, class_type = P.CenterObject.SOFTMAX):
    center_object_loss_param = {
        'loc_weight': loc_weight,
        'num_class': num_classes,
        'share_location': share_location,
        'net_height': net_height,
        'net_width': net_width,
        'ignore_thresh': ignore_thresh,
        'bias_scale': bias_scale,
        'low_bbox_scale': low_bbox_scale,
        'up_bbox_scale': up_bbox_scale,
        'class_type': class_type,
        'bias_num': 1,
    }
    loss_param = {
        'normalization': normalization_mode,
    }
    name = 'CenterGridLoss_{}'.format(stageidx)
    net[name] = L.CenterGridLoss(*from_layers, center_object_loss_param = center_object_loss_param,
                                 loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                                 propagate_down=[True, False])


def CenterGridObjectDetect(net, from_layers = [], bias_scale = [], down_ratio = [], num_classes = 2,
                           ignore_thresh = 0.3,  keep_top_k = 200,
                           class_type = P.DetectionOutput.SOFTMAX, 
                           share_location = True, confidence_threshold = 0.15):
    det_out_param = {
        'num_classes': num_classes,
        'share_location': share_location,
        'keep_top_k': keep_top_k,
        'confidence_threshold': confidence_threshold,
        'class_type': class_type,
        'bias_scale': bias_scale,
        'down_ratio': down_ratio,
    }
    net.detection_out = L.CenterGridOutput(*from_layers, detection_output_param=det_out_param, 
                                                include=dict(phase=caffe_pb2.Phase.Value('TEST')))


def CenterGridMobilenetV2Body(net, from_layer, Use_BN = True, use_global_stats= False, **bn_param):
    assert from_layer in net.keys()
    index = 0
    feature_stride = [4, 8, 16, 32]
    accum_stride = 1
    pre_stride = 1
    LayerList_Name = []
    LayerList_Output = []
    LayerFilters = []
    out_layer = "conv_{}".format(index)
    ConvBNLayer(net, from_layer, out_layer, use_bn=Use_BN, use_relu=True,
                num_output= 32, kernel_size=3, pad=1, stride = 2, use_scale = True,
                use_global_stats= use_global_stats,
                **bn_param)
    accum_stride *= 2
    pre_channels= 32
    Inverted_residual_setting = [[1, 16, 1, 1],
                                 [6, 24, 2, 2],
                                 [6, 32, 3, 2],
                                 [6, 64, 4, 2],
                                 [6, 96, 3, 1],
                                 [6, 160, 3, 2],
                                 [6, 320, 1, 1]]
    for _, (t, c, n, s) in enumerate(Inverted_residual_setting):
        accum_stride *= s
        if n > 1:
            if s == 2:
                layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, Use_BN = True, 
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                out_layer = layer_name
                strides = 1
                for id in range(n - 1):
                    layer_name = MBottleConvBlock(net, out_layer, index, id + 1, c, strides, t, Use_BN = True, 
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                    out_layer = layer_name
            elif s == 1:
                Project_Layer = out_layer
                out_layer= "Conv_project_{}_{}".format(pre_channels, c)
                ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = True, 
                num_output= c, kernel_size= 3, pad= 1, stride= 1,
                lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
                for id in range(n):
                    layer_name = MBottleConvBlock(net, out_layer, index, id, c, s, t, Use_BN = True, 
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                    out_layer = layer_name
        elif n == 1:
            assert s == 1
            Project_Layer = out_layer
            out_layer= "Conv_project_{}_{}".format(pre_channels, c)
            ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = True, 
                        num_output= c, kernel_size= 3, pad= 1, stride= 1,
                        lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
            layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, Use_BN = True, 
                                                        Use_scale = True,use_global_stats= use_global_stats, **bn_param)
            out_layer = layer_name
        if accum_stride in feature_stride:
            if accum_stride != pre_stride:
                LayerList_Name.append(out_layer)
                LayerFilters.append(c)
            elif accum_stride == pre_stride:
                LayerList_Name[len(LayerList_Name) - 1] = out_layer
                LayerFilters[len(LayerFilters) - 1] = c
            pre_stride = accum_stride
        index += 1
        pre_channels = c
    assert len(LayerList_Name) == len(feature_stride)
    net_last_layer = net.keys()[-1]
    out_layer = "conv_1_project/DepthWise"
    ConvBNLayer(net, net_last_layer, out_layer, use_bn = True, use_relu = True, 
                num_output= 320, kernel_size= 3, pad= 1, stride= 2, group= 320,
                lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
    net_last_layer = out_layer
    out_layer = "conv_1_project/linear"
    ConvBNLayer(net, net_last_layer, out_layer, use_bn = True, use_relu = True, 
                num_output= 320, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
    for index in range(len(feature_stride)):
        #Deconv_layer scale up 2x2_s2
        channel_stage = LayerFilters[len(LayerFilters) - index - 1]
        net_last_layer = out_layer
        Reconnect_layer_one = "Deconv_Scale_Up_Stage_{}".format(channel_stage)
        ConvBNLayer(net, net_last_layer, Reconnect_layer_one, use_bn= True, use_relu = False, 
            num_output= channel_stage, kernel_size= 2, pad= 0, stride= 2,
            lr_mult=1, Use_DeConv= True, use_scale= True, use_global_stats= use_global_stats)
        
        # conv_layer linear 1x1
        net_last_layer= LayerList_Name[len(feature_stride) - index - 1]
        Reconnect_layer_two= "{}_linear".format(LayerList_Name[len(feature_stride) - index - 1])
        ConvBNLayer(net, net_last_layer, Reconnect_layer_two, use_bn= True, use_relu = False, 
            num_output= channel_stage, kernel_size= 1, pad= 0, stride= 1,
            lr_mult=1, use_scale= True, use_global_stats= use_global_stats)
        
        # eltwise_sum layer
        out_layer, detect_layer = ResConnectBlock(net, Reconnect_layer_one, Reconnect_layer_two, channel_stage, use_global_stats=use_global_stats)
        LayerList_Output.append(detect_layer)
    return net, LayerList_Output


def efficientNetBody(net, from_layer, width_coefficient, depth_coefficient, Use_BN = True, use_global_stats= False, **bn_param):
    assert from_layer in net.keys()
    index = 0
    feature_stride = [8, 16, 32]
    accum_stride = 1
    pre_stride = 1
    LayerList_Name = []
    LayerFilters = []
    out_layer = "conv_{}".format(index)
    Param_width_channel= round_filters(32, width_coefficient)
    ConvBNLayer(net, from_layer, out_layer, use_bn=Use_BN, use_relu=True,
                num_output= Param_width_channel, kernel_size=3, pad=1, stride = 2, use_scale = True,
                use_global_stats= use_global_stats,
                **bn_param)
    accum_stride *= 2
    pre_channels= Param_width_channel
                                 #e  c   r  s  k
    Inverted_residual_setting = [[1, 16, 1, 1, 3],
                                 [6, 24, 2, 2, 3],
                                 [6, 40, 2, 2, 5],
                                 [6, 80, 3, 2, 3],
                                 [6, 112, 3, 1, 5],
                                 [6, 192, 4, 2, 5],
                                 [6, 320, 1, 1, 3]]
    '''
    _DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    '''
    for _, (t, c, n, s, k) in enumerate(Inverted_residual_setting):
        accum_stride *= s
        c = round_filters(c, width_coefficient)
        n = round_repeats(n, depth_coefficient)
        if n > 1:
            if s == 2:
                layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, kernel_size= k, Use_BN = True, 
                                                        Use_scale = True, use_global_stats= use_global_stats, Use_SE=True,  **bn_param)
                out_layer = layer_name
                strides = 1
                for id in range(n - 1):
                    layer_name = MBottleConvBlock(net, out_layer, index, id + 1, c, strides, t, kernel_size= k, Use_BN = True, 
                                                        Use_scale = True, use_global_stats= use_global_stats, Use_SE=True, **bn_param)
                    out_layer = layer_name
            elif s == 1:
                Project_Layer = out_layer
                out_layer= "Conv_project_{}_{}".format(pre_channels, c)
                ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = True, 
                num_output= c, kernel_size= 3, pad= 1, stride= 1,
                lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
                for id in range(n):
                    layer_name = MBottleConvBlock(net, out_layer, index, id, c, s, t, kernel_size= k, Use_BN = True, 
                                                        Use_scale = True, use_global_stats= use_global_stats, Use_SE=True, **bn_param)
                    out_layer = layer_name
        elif n == 1:
            assert s == 1
            Project_Layer = out_layer
            out_layer= "Conv_project_{}_{}".format(pre_channels, c)
            ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = True, 
                        num_output= c, kernel_size= 3, pad= 1, stride= 1,
                        lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
            layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, kernel_size= k, Use_BN = True, 
                                                        Use_scale = True,use_global_stats= use_global_stats, Use_SE=True, **bn_param)
            out_layer = layer_name
        if accum_stride in feature_stride:
            if accum_stride != pre_stride:
                LayerList_Name.append(out_layer)
                LayerFilters.append(c)
            elif accum_stride == pre_stride:
                LayerList_Name[len(LayerList_Name) - 1] = out_layer
                LayerFilters[len(LayerFilters) - 1] = c
            pre_stride = accum_stride
        index += 1
        pre_channels = c
    assert len(LayerList_Name) == len(feature_stride)
    return net, LayerList_Name