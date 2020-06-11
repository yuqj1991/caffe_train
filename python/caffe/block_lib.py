import os,sys,logging
try:
    caffe_root = '../../../caffe_train/'
    sys.path.insert(0, caffe_root + 'python')
    import caffe
except ImportError:
    logging.fatal("block_lib.py Cannot find caffe!")
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from caffe.utils import *
import math

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
        out_layer_depthswise = "conv_{}_{}/{}".format(id, repeated_num, "conv")
        ConvBNLayer(net, from_layer, out_layer_depthswise, use_bn=Use_BN, use_relu = use_relu, use_swish= use_swish,
                    num_output = input_channels * expansion_factor, kernel_size=kernel_size, pad=pad,
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
        ConvBNLayer(net, from_layer, out_layer_depthswise, use_bn=Use_BN, use_relu = use_relu, use_swish= use_swish,
                    num_output = input_channels * expansion_factor, kernel_size=kernel_size, pad=0, stride = strides,
                    use_scale = Use_scale, use_global_stats= use_global_stats, **bn_param)
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


def CenterFaceMobilenetV2Body(net, from_layer, Use_BN = True, use_global_stats= False, 
								Inverted_residual_setting = [[1, 16, 1, 1],
                                 [6, 24, 2, 2],
                                 [6, 32, 3, 2],
                                 [6, 64, 4, 2],
                                 [6, 96, 3, 1],
                                 [6, 160, 3, 2],
                                 [6, 320, 1, 1]],
								detect_num=4, num_class= 1, use_branch = False, **bn_param):
    assert from_layer in net.keys()
    index = 0
    feature_stride = [4, 8, 16, 32]
    accum_stride = 1
    pre_stride = 1
    LayerList_Name = []
    LayerFilters = []
    out_layer = "conv_{}".format(index)
    ConvBNLayer(net, from_layer, out_layer, use_bn=Use_BN, use_relu=True,
                num_output= 32, kernel_size=3, pad=1, stride = 2, use_scale = True,
                use_global_stats= use_global_stats,
                **bn_param)
    accum_stride *= 2
    pre_channels= 32
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
    fpn_out_channels = 24
    out_layer = "conv_1_project/linear"
    ConvBNLayer(net, net_last_layer, out_layer, use_bn = True, use_relu = True, 
                num_output= fpn_out_channels, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
    net_last_layer = out_layer
    for index in range(len(feature_stride) - 1):
        #Deconv_layer scale up 2x2_s2
        channel_stage = LayerFilters[len(LayerFilters) - index - 1]
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
        net_last_layer = detect_layer

    last_conv_layer = "last_conv_3x3_layer"
    ConvBNLayer(net, net_last_layer, last_conv_layer, use_bn = True, use_relu = True, 
                num_output= fpn_out_channels, kernel_size= 3, pad= 1, stride= 1,
                lr_mult=1, use_scale=True, use_global_stats= use_global_stats)
    net_last_layer = last_conv_layer
    ### class prediction layer
    Class_out = "Class_out_1x1"
    ConvBNLayer(net, net_last_layer, Class_out, use_bn= False, 
                use_swish= False, use_relu = False, 
                num_output= num_class, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale= False, use_global_stats= use_global_stats)

    ### Box loc prediction layer
    Box_offset_out = "Box_out_offset_1x1"
    ConvBNLayer(net, net_last_layer, Box_offset_out, use_bn= False, 
                use_swish= False, use_relu = False, 
                num_output= 2, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale= False, use_global_stats= use_global_stats)
    Box_wh_out = "Box_out_wh_1x1"
    ConvBNLayer(net, net_last_layer, Box_wh_out, use_bn= False, 
                use_swish= False, use_relu = False, 
                num_output= 2, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale= False, use_global_stats= use_global_stats)
    Box_out = []
    Box_out.append(net[Box_offset_out])
    Box_out.append(net[Box_wh_out])
    lm_out = ''
    if (detect_num - 2 - 2) > 0:
        assert(detect_num == 14)
        lm_out = "landmarks_out_1x1"
        ConvBNLayer(net, net_last_layer, lm_out, use_bn= False, 
                use_swish= False, use_relu = False, 
                num_output= detect_num - 4, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale= False, use_global_stats= use_global_stats)
        Box_out.append(net[lm_out])
    Concat_out = "Box_out_1x1"
    if use_branch:
        Concat_out = []
        Concat_out.append(Box_offset_out)
        Concat_out.append(Box_wh_out)
        if (detect_num - 2 - 2) > 0:
            Concat_out.append(lm_out)
        return net, Class_out, Concat_out
    net[Concat_out] = L.Concat(*Box_out, axis=1)
    return net, Class_out, Concat_out

def CenterGridfaceBody(net, from_layer, Use_BN = True, 
								use_global_stats= False, Inverted_residual_setting = [[1, 16, 1, 1],
                                 [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],[6, 96, 3, 1], 
                                 [6, 160, 3, 2], [6, 320, 1, 1]],
                                 feature_stride = [4, 8, 16, 32],
                                 Fpn= True, biFpn = True, fpn_out_channels = 24,
                                 top_out_channels = 320,
                                 detector_num = 6, num_class = 2, **bn_param):
    assert from_layer in net.keys()
    index = 0
    accum_stride = 1
    LayerList_Name = []
    LayerList_Output = []
    LayerFilters = []
    moudle_output = []
    out_layer = "conv_{}".format(index)
    ConvBNLayer(net, from_layer, out_layer, use_bn=Use_BN, use_relu=True,
                num_output= 32, kernel_size=3, pad=0, stride = 2, use_scale = True,
                use_global_stats= use_global_stats,
                **bn_param)
    accum_stride *= 2
    pre_channels= 32
    for _, (t, c, n, s) in enumerate(Inverted_residual_setting):
        accum_stride *= s
        if n > 1:
            if s == 2:
                layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, pre_channels, Use_BN = Use_BN, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = Use_BN, use_global_stats= use_global_stats, **bn_param)
                out_layer = layer_name
                pre_channels = c
                strides = 1
                moudle_output.append({"layer": out_layer, "stride": accum_stride})
                for id in range(n - 1):
                    layer_name = MBottleConvBlock(net, out_layer, index, id + 1, c, strides, t, pre_channels, Use_BN = Use_BN, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = Use_BN, use_global_stats= use_global_stats, **bn_param)
                    out_layer = layer_name
                    pre_channels = c
                    moudle_output.append({"layer": out_layer, "stride": accum_stride, "c": c})
            elif s == 1:
                Project_Layer = out_layer
                out_layer= "Conv_project_{}_{}".format(pre_channels, c)
                ConvBNLayer(net, Project_Layer, out_layer, use_bn = Use_BN, use_relu = True, 
                            use_swish= False,
                            num_output= c, kernel_size= 3, pad= 1, stride= 1,
                            lr_mult=1, use_scale=Use_BN, use_global_stats= use_global_stats)
                pre_channels = c
                for id in range(n):
                    layer_name = MBottleConvBlock(net, out_layer, index, id, c, s, t, pre_channels, Use_BN = Use_BN, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = Use_BN, use_global_stats= use_global_stats, **bn_param)
                    out_layer = layer_name
                    pre_channels = c
                    moudle_output.append({"layer": out_layer, "stride": accum_stride, "c": c})
        elif n == 1:
            assert s == 1
            layer_name = MBottleConvBlock(net, out_layer, index, 0, c, s, t, pre_channels,  Use_BN = Use_BN, 
                                                        use_relu= True, use_swish= False,
                                                        Use_scale = Use_BN, use_global_stats= use_global_stats, **bn_param)
            pre_channels = c
            out_layer = layer_name
            moudle_output.append({"layer": out_layer, "stride": accum_stride, "c": c})
        index += 1
    parse_stride = []
    num = 1
    for id in range(1, len(feature_stride)):
        if feature_stride[id - 1] != feature_stride[id]:
            parse_stride.append({"stride": feature_stride[id - 1], "num": num})
            num = 1
            if(id == len(feature_stride) - 1):
                parse_stride.append({"stride": feature_stride[id], "num": num})
        else:
            num += 1
            if(id == len(feature_stride) - 1):
                parse_stride.append({"stride": feature_stride[id], "num": num})
    parse_out_stride = []
    num = 1
    for id in range(1, len(moudle_output)):
        if moudle_output[id - 1]["stride"] != moudle_output[id]["stride"]:
            parse_out_stride.append({"stride": moudle_output[id - 1]["stride"], "num": num, "idx": id - 1})
            num = 1
            if(id == len(moudle_output) - 1):
                parse_out_stride.append({"stride": moudle_output[id]["stride"], "num": num, "idx": id})
        else:
            num += 1
            if(id == len(moudle_output) - 1):
                parse_out_stride.append({"stride": moudle_output[id]["stride"], "num": num, "idx": id})

    for id in range(len(parse_stride)):
        stride = parse_stride[id]['stride']
        num = parse_stride[id]['num']
        for _, out_stride in enumerate(parse_out_stride):
            if stride == out_stride['stride']:
                idx = out_stride['idx']
                for index in range(idx + 1 - num,idx + 1):
                    LayerList_Name.append(moudle_output[index]['layer'])
                    LayerFilters.append(moudle_output[index]['c'])
    print(LayerList_Name)
    assert len(LayerList_Name) == len(feature_stride)
    for index, detect_layer in enumerate(LayerList_Name):
        ch_stage = LayerFilters[index]
        conv_out = "conv_3x3_out_{}_{}".format(ch_stage, index)
        ConvBNLayer(net, detect_layer, conv_out, use_bn= Use_BN, 
                use_swish= False, use_relu = True, 
                num_output= fpn_out_channels, kernel_size= 3, pad= 1, stride= 1,
                lr_mult=1, use_scale= Use_BN, use_global_stats= False)
        detectionBox_conv_layer = "Det_1x1_out_{}_{}".format(ch_stage, index)
        ConvBNLayer(net, conv_out, detectionBox_conv_layer, use_bn= False, 
                use_swish= False, use_relu = False, 
                num_output= detector_num, kernel_size= 1, pad= 0, stride= 1,
                lr_mult=1, use_scale= False, use_global_stats= False)
        LayerList_Output.append(detectionBox_conv_layer)
    return net, LayerList_Output