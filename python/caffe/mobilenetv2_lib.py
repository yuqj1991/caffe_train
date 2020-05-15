import os,sys,logging
try:
    caffe_root = '../../../caffe_train/'
    sys.path.insert(0, caffe_root + 'python')
    import caffe
except ImportError:
    logging.fatal("mobilenetv2_lib.py Cannot find caffe!")
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from caffe.utils import *
import math


def SEMoudleBlock(net, from_layer, channels, layerPrefix= '', ratio = 0.2, use_swish = True):
    assert from_layer in net.keys()
    Global_poolingName= "{}_GloabalPool".format(layerPrefix)
    net[Global_poolingName] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
    Full_inproductName_Project = "{}_inProduct_Project".format(layerPrefix)
    net[Full_inproductName_Project] = L.InnerProduct(net[Global_poolingName], num_output = int(channels * ratio))
    if use_swish:
        Swish_Name = "{}_Swish".format(layerPrefix)
        net[Swish_Name] = L.Swish(net[Full_inproductName_Project], in_place=True)
    else:
        Swish_Name = "{}_Relu6".format(layerPrefix)
        net[Swish_Name] = L.ReLU6(net[Full_inproductName_Project], in_place=True)
    Full_inproductName_expand = "{}_inProduct_Expand".format(layerPrefix)
    net[Full_inproductName_expand] = L.InnerProduct(net[Swish_Name], num_output = channels)
    Sigmoid_Name = "{}_Sigmoid".format(layerPrefix)
    net[Sigmoid_Name] = L.Sigmoid(net[Full_inproductName_expand], in_place= True)
    Scale_name = "{}_AttentionScale".format(layerPrefix)
    net[Scale_name] = L.AttentionScale(net[from_layer], net[Sigmoid_Name])
    return Scale_name


def BiFPNBlock(net, from_layers= [], image_size = 640, min_level = 3, max_level = 7, fpn_cell_repeats = 3, 
                    fpn_out_channels = 88, use_global_stats = True, use_relu = False, use_swish = True,
                    apply_bn=True, is_training=True, conv_after_downsample=False, 
                    use_nearest_resize=False, pooling_type= None):
    assert len(from_layers) > 0
    for i, layer in enumerate(from_layers):
        if not isinstance(layer, dict):
            raise ValueError("layer must be dict")
        assert layer['layer'] in net.keys()
    feat_sizes = get_feat_sizes(image_size, max_level)
    num_levels = max_level - min_level + 1
    feats = []
    # get FPN Layer
    for i in range(num_levels):
        if i < len(from_layers):
            feats.append(from_layers[i])
            if not isinstance(feats[i], dict):
                raise ValueError("feats[{}] must be dict".format(i))
        else:
            if isinstance(feats[-1], dict):
                feats.append(resample_feature_map(net, feats[-1], use_global_stats= use_global_stats, 
                                use_relu= False, 
                                target_height= (feats[-1]['height'] - 1) // 2 + 1, 
                                target_width= (feats[-1]["width"]  - 1) // 2 + 1,
                                target_channels= fpn_out_channels, 
                                layerPrefix = '{}_base'.format(i), 
                                apply_bn= apply_bn, is_training= is_training, conv_after_downsample=conv_after_downsample, 
                                use_nearest_resize= use_nearest_resize, pooling_type= pooling_type))
            else:
                raise ValueError("feats[{}] must be dict type".format(i - 1))
    verify_feats_size(net= net, feats= feats, feat_sizes= feat_sizes, min_level= min_level, max_level= max_level)

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
        new_feats = BuildBiFPNLayer(net= net, feats= feats, feat_sizes= feat_sizes, fpn_nodes= pnodes, 
                                        layerPrefix = '{}_BiFPN'.format(repeate_idx),
                                        fpn_out_filters= fpn_out_channels, min_level = min_level, max_level = max_level, 
                                        use_global_stats = use_global_stats, use_relu = use_relu, use_swish= use_swish, concat_method= "fast_attention", 
                                        apply_bn=apply_bn, is_training=is_training, conv_after_downsample=conv_after_downsample,
                                        separable_conv = True, use_nearest_resize=use_nearest_resize, pooling_type= pooling_type)
        feats = [new_feats[level] for level in range(min_level, max_level + 1)]
        verify_feats_size(net= net, feats= feats, feat_sizes= feat_sizes, min_level= min_level, max_level= max_level)
    return new_feats
        


def MBottleConvBlock(net, from_layer, id, repeated_num, fileter_channels, strides, expansion_factor,
                        input_channels,
                        kernel_size= 3, Use_BN = True, Use_scale = True, 
                        use_global_stats= False, Use_SE= False, use_relu = False, 
                        use_swish= False, **bn_param):
    if kernel_size == 3:
        pad = 1
    elif kernel_size == 5:
        pad = 2
    if strides == 1:
        out_layer_expand = "conv_{}_{}/{}".format(id, repeated_num, "expand")
        ConvBNLayer(net, from_layer, out_layer_expand, use_bn=Use_BN, use_relu = use_relu, use_swish= use_swish,
                    num_output = input_channels * expansion_factor, kernel_size=1, 
                    pad=0, stride = strides, use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        if Use_SE:
            out_layer_depthswise = "conv_{}_{}/{}".format(id, repeated_num, "depthwise")
            ConvBNLayer(net, out_layer_expand, out_layer_depthswise, use_bn=Use_BN, use_relu = False, use_swish= False,
                        num_output = input_channels * expansion_factor, kernel_size=kernel_size, pad=pad, 
                        group= input_channels * expansion_factor,
                        stride = strides, use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
            out_layer = out_layer_depthswise
            out_layer = SEMoudleBlock(net, from_layer= out_layer, channels= input_channels * expansion_factor, 
                                        layerPrefix= 'SE_{}_{}/{}'.format(id, repeated_num, 'attention'), ratio = 0.25, 
                                        use_swish= use_swish) 
        else:
            out_layer_depthswise = "conv_{}_{}/{}".format(id, repeated_num, "depthwise")
            ConvBNLayer(net, out_layer_expand, out_layer_depthswise, use_bn=Use_BN, use_relu = use_relu, use_swish= use_swish,
                        num_output = input_channels * expansion_factor, kernel_size=kernel_size, pad=pad, 
                        group= input_channels * expansion_factor,
                        stride = strides, use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
            out_layer = out_layer_depthswise
        out_layer_projects = "conv_{}_{}/{}".format(id, repeated_num, "linear")
        ConvBNLayer(net, out_layer, out_layer_projects, use_bn=Use_BN, use_relu = False, use_swish= False,
                    num_output = fileter_channels, kernel_size=1, pad=0, stride = strides, 
                    use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        if input_channels == fileter_channels:
            res_name = 'Res_Sum_{}_{}'.format(id, repeated_num)
            net[res_name] = L.Eltwise(net[from_layer], net[out_layer_projects])
            return res_name
        else:
            Relu_layer = "conv_{}_{}/{}_relu6".format(id, repeated_num, "linear")
            net[Relu_layer] = L.ReLU6(net[out_layer_projects], in_place = True)
            return Relu_layer
    elif strides == 2:
        out_layer_expand = "conv_{}_{}/{}".format(id, repeated_num, "expand")
        ConvBNLayer(net, from_layer, out_layer_expand, use_bn=Use_BN, use_relu = use_relu, use_swish= use_swish,
                    num_output = input_channels * expansion_factor, kernel_size=1, pad=0, stride = 1, 
                    use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        out_layer_depthswise = "conv_{}_{}/{}".format(id, repeated_num, "depthwise")
        ConvBNLayer(net, out_layer_expand, out_layer_depthswise, use_bn=Use_BN, use_relu = use_relu, use_swish= use_swish,
                    num_output = input_channels * expansion_factor, kernel_size=kernel_size, pad=pad, stride = strides, 
                    group= input_channels * expansion_factor, use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
        out_layer_projects = "conv_{}_{}/{}".format(id, repeated_num, "linear")
        ConvBNLayer(net, out_layer_depthswise, out_layer_projects, use_bn=Use_BN, use_relu = False, use_swish= False,
                    num_output = fileter_channels, kernel_size=1, pad=0, stride = 1, use_scale = Use_scale
                    , use_global_stats= use_global_stats,
                    **bn_param)
        return out_layer_projects


def ResConnectBlock(net, from_layer_one, from_layer_two, stage_idx,  use_relu, layerPrefix):
    res_name = "{}_stage_{}".format(layerPrefix, stage_idx)
    net[res_name] = L.Eltwise(net[from_layer_one], net[from_layer_two], operation = P.Eltwise.SUM)
    if use_relu:
        Relu_layer = "{}_stage_{}/Relu6".format(layerPrefix, stage_idx)
        net[Relu_layer] = L.ReLU6(net[res_name], in_place = True)
        out_layer = Relu_layer
    else:
        out_layer = res_name
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

def CenterFaceObjectDetect(net, from_layers = [],  num_classes = 2,
                           ignore_thresh = 0.3,  keep_top_k = 200,
                           share_location = True, confidence_threshold = 0.15):
    det_out_param = {
        'num_classes': num_classes,
        'share_location': share_location,
        'keep_top_k': keep_top_k,
        'confidence_threshold': confidence_threshold,
    }
    net.detection_out = L.CenternetDetectionOutput(*from_layers, detection_output_param=det_out_param, 
                                                include=dict(phase=caffe_pb2.Phase.Value('TEST')))


def CenterFaceObjectLoss(net, stageidx, from_layers = [], loc_loss_type = P.CenterObject.SMOOTH_L1,
                         normalization_mode = P.Loss.VALID, num_classes= 1, loc_weight = 1.0, 
                         share_location = True, ignore_thresh = 0.3, class_type = P.CenterObject.FOCALSIGMOID):
    center_object_loss_param = {
        'loc_weight': loc_weight,
        'num_class': num_classes,
        'loc_loss_type': loc_loss_type,
        'conf_loss_type': class_type,
        'share_location': share_location
    }
    loss_param = {
        'normalization': normalization_mode,
    }
    name = 'CenterFaceLoss_{}'.format(stageidx)
    net[name] = L.CenterObjectLoss(*from_layers, center_object_loss_param = center_object_loss_param,
                                 loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                                 propagate_down=[True, True, False])


def CenterFaceMobilenetV2Body(net, from_layer, Use_BN = True, use_global_stats= False, **bn_param):
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
                layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, pre_channels, Use_BN = True, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                out_layer = layer_name
                pre_channels = c
                strides = 1
                for id in range(n - 1):
                    layer_name = MBottleConvBlock(net, out_layer, index, id + 1, c, strides, t, pre_channels, Use_BN = True, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                    out_layer = layer_name
                    pre_channels = c
            elif s == 1:
                Project_Layer = out_layer
                out_layer= "Conv_project_{}_{}".format(pre_channels, c)
                ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = True, 
                            use_swish= False,
                            num_output= c, kernel_size= 3, pad= 1, stride= 1,
                            lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
                pre_channels = c
                for id in range(n):
                    layer_name = MBottleConvBlock(net, out_layer, index, id, c, s, t, pre_channels, Use_BN = True, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                    out_layer = layer_name
                    pre_channels = c
        elif n == 1:
            assert s == 1
            '''
            Project_Layer = out_layer
            out_layer= "DepethWiseConv_{}_{}".format(pre_channels, c)
            ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = True, 
                        use_swish= False, group= pre_channels,
                        num_output= pre_channels, kernel_size= 3, pad= 1, stride= 1,
                        lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
            Project_Layer = out_layer
            out_layer= "PointWiseConv_{}_{}".format(pre_channels, c)
            ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = False, 
                        use_swish= False, 
                        num_output= c, kernel_size= 1, pad= 0, stride= 1,
                        lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
            pre_channels = c
            '''
            layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, pre_channels,  Use_BN = True, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = True,use_global_stats= use_global_stats, **bn_param)
            pre_channels = c
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
    assert len(LayerList_Name) == len(feature_stride)
    net_last_layer = net.keys()[-1]
    out_layer = "conv_1_project/linear"
    ConvBNLayer(net, net_last_layer, out_layer, use_bn = True, use_relu = True, 
                num_output= 128, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
    fpn_out_channels = 24
    for index in range(len(feature_stride) - 1):
        #Deconv_layer scale up 2x2_s2
        channel_stage = LayerFilters[len(LayerFilters) - index - 1]
        net_last_layer = out_layer
        Reconnect_layer_one = "Deconv_Scale_Up_Stage_{}".format(channel_stage)
        ConvBNLayer(net, net_last_layer, Reconnect_layer_one, use_bn= True, use_relu = False, 
            num_output= fpn_out_channels, kernel_size= 2, pad= 0, stride= 2,
            lr_mult=1, Use_DeConv= True, use_scale= True, use_global_stats= use_global_stats)

        Reconnect_layer_two = "{}_linear_Conv".format(index)
        net_last_layer = LayerList_Name[len(LayerFilters) - index - 1 - 1]
        ConvBNLayer(net, net_last_layer, Reconnect_layer_two, use_bn= True, 
                use_swish= False, use_relu = False, 
                num_output= fpn_out_channels, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale= True, use_global_stats= use_global_stats)
        
        # eltwise_sum layer
        _, detect_layer = ResConnectBlock(net, Reconnect_layer_one, Reconnect_layer_two, channel_stage, 
                                            use_relu=True, layerPrefix = "Res_conv_linear")
        out_layer = detect_layer

    ### class prediction layer
    Class_out = "Class_out_1x1"
    ConvBNLayer(net, out_layer, Class_out, use_bn= False, 
                use_swish= False, use_relu = False, 
                num_output= 1, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale= False, use_global_stats= use_global_stats)

    ### Box loc prediction layer
    Box_out = "Box_out_1x1"
    ConvBNLayer(net, out_layer, Box_out, use_bn= False, 
                use_swish= False, use_relu = False, 
                num_output= 4, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale= False, use_global_stats= use_global_stats)
    return net, Class_out, Box_out

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
                layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, pre_channels, Use_BN = True, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                out_layer = layer_name
                pre_channels = c
                strides = 1
                for id in range(n - 1):
                    layer_name = MBottleConvBlock(net, out_layer, index, id + 1, c, strides, t, pre_channels, Use_BN = True, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                    out_layer = layer_name
                    pre_channels = c
            elif s == 1:
                Project_Layer = out_layer
                out_layer= "Conv_project_{}_{}".format(pre_channels, c)
                ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = True, 
                            use_swish= False,
                            num_output= c, kernel_size= 3, pad= 1, stride= 1,
                            lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
                pre_channels = c
                for id in range(n):
                    layer_name = MBottleConvBlock(net, out_layer, index, id, c, s, t, pre_channels, Use_BN = True, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = True, use_global_stats= use_global_stats, **bn_param)
                    out_layer = layer_name
                    pre_channels = c
        elif n == 1:
            assert s == 1
            '''
            Project_Layer = out_layer
            out_layer= "Conv_project_{}_{}".format(pre_channels, c)
            ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, use_relu = False, 
                        use_swish= False,
                        num_output= c, kernel_size= 3, pad= 1, stride= 1,
                        lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
            pre_channels = c
            '''
            layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, pre_channels,  Use_BN = True, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = True,use_global_stats= use_global_stats, **bn_param)
            pre_channels = c
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
    
    fpn_out_channels = 128
    for idx, feature_layer in enumerate(LayerList_Name):
        ConvLayer = "DepthWise_conv_{}_{}".format(idx, feature_layer)
        channel_stage = LayerFilters[idx]
        ConvBNLayer(net, feature_layer, ConvLayer, use_bn= True, use_relu = True, 
            use_swish= False, group= channel_stage,
            num_output= channel_stage, kernel_size= 3, pad= 1, stride= 1,
            lr_mult=1, use_scale= True, use_global_stats= use_global_stats)
        PointLayer = "linear_conv_{}_{}".format(idx, feature_layer)
        ConvBNLayer(net, ConvLayer, PointLayer, use_bn= True, use_relu = False, use_swish= False,
            num_output= fpn_out_channels, kernel_size= 1, pad= 0, stride= 1,
            lr_mult=1, use_scale= True, use_global_stats= use_global_stats)
        LayerList_Name[idx] = PointLayer

    for index in range(len(feature_stride)):
        #Deconv_layer scale up 2x2_s2
        channel_stage = LayerFilters[len(LayerFilters) - index - 1]
        net_last_layer = out_layer
        Reconnect_layer_one = "Deconv_Scale_Up_Stage_{}".format(channel_stage)
        ConvBNLayer(net, net_last_layer, Reconnect_layer_one, use_bn= True, use_relu = False, 
            num_output= fpn_out_channels, kernel_size= 2, pad= 0, stride= 2,
            lr_mult=1, Use_DeConv= True, use_scale= True, use_global_stats= use_global_stats)
        
        if index != 3:
            Res_Layer_one = "FPN_con_{}".format(index)
            net_last_layer = LayerList_Name[len(feature_stride) - index - 2]
            ConvBNLayer(net, net_last_layer, Res_Layer_one, use_bn= True, 
                use_swish= False, use_relu = False, 
                num_output= fpn_out_channels, kernel_size= 3, pad= 1, stride= 2,
                lr_mult=1, use_scale= True, use_global_stats= use_global_stats)
            Res_Layer_two = LayerList_Name[len(feature_stride) - index - 1]
            _, Reconnect_layer_two = ResConnectBlock(net, Res_Layer_one, Res_Layer_two, index, use_relu=False, layerPrefix = "FPN_linear_".format(index))
        else:
            Reconnect_layer_two= LayerList_Name[len(feature_stride) - index - 1]
        # eltwise_sum layer
        out_layer, detect_layer = ResConnectBlock(net, Reconnect_layer_one, Reconnect_layer_two, channel_stage, use_relu=False, layerPrefix = "Dectction")
        LayerList_Output.append(detect_layer)
    return net, LayerList_Output


def efficientNetBody(net, from_layer, width_coefficient, depth_coefficient, Use_BN = True, use_global_stats= False, 
                            use_relu = False, use_swish= True, inputHeight= 640, inputWidth = 640, **bn_param):
    assert from_layer in net.keys()
    if use_relu and use_swish:
        raise ValueError("the use_relu and use_swish should not be true at the same time")
    if not use_relu and not use_swish:
        raise ValueError("the use_relu and use_swish should not be false at the same time")
    index = 0
    feature_stride = [8, 16, 32]
    accum_stride = 1
    pre_stride = 1
    LayerList_Name = []
    LayerFilters = []
    LayerShapes = []
    out_layer = "conv_{}".format(index)
    Param_width_channel= round_filters(32, width_coefficient)
    ConvBNLayer(net, from_layer, out_layer, use_bn=Use_BN, use_relu=use_relu, use_swish= use_swish,
                num_output= Param_width_channel, kernel_size=3, pad=1, stride = 2, use_scale = True,
                use_global_stats= use_global_stats,
                **bn_param)
    layer_height, layer_width = get_layer_shape(False, inputHeight, inputWidth, 3, 1, 2, "conv")
    current_height, current_width= layer_height, layer_width
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
                layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, pre_channels,  kernel_size= k, Use_BN = True, 
                                                        use_relu= use_relu, use_swish= use_swish,
                                                        Use_scale = True, use_global_stats= use_global_stats, Use_SE=True,  **bn_param)
                out_layer = layer_name
                pre_channels= c
                if k == 3:
                    pad = 1
                elif k == 5:
                    pad = 2
                layer_height, layer_width = get_layer_shape(False, current_height, current_width, k, pad, s, "conv")
                current_height, current_width= layer_height, layer_width
                strides = 1
                for id in range(n - 1):
                    layer_name = MBottleConvBlock(net, out_layer, index, id + 1, c, strides, t, pre_channels, kernel_size= k, Use_BN = True, 
                                                        use_relu=use_relu, use_swish= use_swish,
                                                        Use_scale = True, use_global_stats= use_global_stats, Use_SE=True, **bn_param)
                    if k == 3:
                        pad = 1
                    elif k == 5:
                        pad = 2
                    out_layer = layer_name
                    pre_channels= c
                    layer_height, layer_width = get_layer_shape(False, current_height, current_width, k, pad, strides, "conv")
                    current_height, current_width= layer_height, layer_width
            elif s == 1:
                Project_Layer = out_layer
                out_layer= "Conv_project_{}_{}".format(pre_channels, c)
                ConvBNLayer(net, Project_Layer, out_layer, use_bn = True,
                                use_relu=use_relu, use_swish= use_swish,
                                num_output= c, kernel_size= 3, pad= 1, stride= 1,
                                lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
                pre_channels= c
                layer_height, layer_width = get_layer_shape(False, current_height, current_width, 3, 1, 1, "conv")
                current_height, current_width= layer_height, layer_width
                for id in range(n):
                    layer_name = MBottleConvBlock(net, out_layer, index, id, c, s, t, pre_channels, kernel_size= k, Use_BN = True, 
                                                        use_relu= use_relu, use_swish= use_swish,
                                                        Use_scale = True, use_global_stats= use_global_stats, Use_SE=True, **bn_param)
                    out_layer = layer_name
                    pre_channels= c
                    if k == 3:
                        pad = 1
                    elif k == 5:
                        pad = 2
                    out_layer = layer_name
                    layer_height, layer_width = get_layer_shape(False, current_height, current_width, k, pad, s, "conv")
                    current_height, current_width= layer_height, layer_width
        elif n == 1:
            assert s == 1
            '''
            Project_Layer = out_layer
            out_layer= "Conv_project_{}_{}".format(pre_channels, c)
            ConvBNLayer(net, Project_Layer, out_layer, use_bn = True, 
                        use_relu= False, use_swish= False,
                        num_output= c, kernel_size= 3, pad= 1, stride= 1,
                        lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
            pre_channels= c
            layer_height, layer_width = get_layer_shape(False, current_height, current_width, 3, 1, 1, "conv")
            current_height, current_width= layer_height, layer_width
            '''
            layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, pre_channels, kernel_size= k, Use_BN = True, 
                                                        use_relu= use_relu, use_swish= use_swish,
                                                        Use_scale = True,use_global_stats= use_global_stats, Use_SE=True, **bn_param)
            out_layer = layer_name
            pre_channels= c
            if k == 3:
                pad = 1
            elif k == 5:
                pad = 2
            out_layer = layer_name
            layer_height, layer_width = get_layer_shape(False, current_height, current_width, k, pad, s, "conv")
            current_height, current_width= layer_height, layer_width

        if accum_stride in feature_stride:
            if accum_stride != pre_stride:
                LayerList_Name.append(out_layer)
                LayerFilters.append(c)
                LayerShapes.append({'height': current_height, 'width': current_width, 'channel': c, 'layer': out_layer})
            elif accum_stride == pre_stride:
                LayerList_Name[len(LayerList_Name) - 1] = out_layer
                LayerFilters[len(LayerFilters) - 1] = c
                LayerShapes[len(LayerShapes) - 1] = {'height': current_height, 'width': current_width, 'channel': c, 'layer': out_layer}
            pre_stride = accum_stride
        index += 1
    assert len(LayerList_Name) == len(feature_stride)
    return net, LayerShapes

def efficientDetBody(net, from_layer, width_coefficient, depth_coefficient, Use_BN = True, use_global_stats= False, 
                                use_relu = False, use_swish= True,
                                is_training=True, conv_after_downsample=False, 
                                use_nearest_resize=False, pooling_type= None):
    if use_relu and use_swish:
        raise ValueError("the use_relu and use_swish should not be true at the same time")
    if not use_relu and not use_swish:
        raise ValueError("the use_relu and use_swish should not be false at the same time")
    net, BaseLayer_Shapes = efficientNetBody(net= net, from_layer= from_layer, width_coefficient= width_coefficient, 
                                            depth_coefficient= depth_coefficient, 
                                            Use_BN= Use_BN, use_global_stats= use_global_stats,
                                            use_relu = use_relu, use_swish= use_swish, inputHeight= 640, inputWidth = 640)
    FPNlayer_Name = BiFPNBlock(net= net, from_layers= BaseLayer_Shapes,
                                image_size = 640, min_level = 3, 
                                max_level = 7, fpn_cell_repeats = 3, 
                                fpn_out_channels = 88, use_global_stats = use_global_stats, use_relu= use_relu, use_swish= use_swish,
                                apply_bn=Use_BN, is_training=is_training, conv_after_downsample=conv_after_downsample, 
                                use_nearest_resize= use_nearest_resize, pooling_type= pooling_type)
    '''
    print(FPNlayer_List)
    
    class_out, box_out = Build_class_and_box_outputs(net, FPNlayer_Name, 88, num_classes= 99, min_level= 3, 
                                                            max_level= 7, is_training_bn= is_training)
    print(class_out)
    print("**************************")
    print(box_out)
    '''
    return FPNlayer_Name
    