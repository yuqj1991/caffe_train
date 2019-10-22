python demo_image_det_ccpd.py --model ../prototxt/deep_ccpd_det_deploy_v1.prototxt --weights ../snapshot/deepccpd_res_v1_iter_370000.caffemodel --ccpdmodel ../prototxt/deep_ccpd_rec_deploy_v1.prototxt --ccpdweights ../snapshot/deepccpd_rec_v1_iter_1560000.caffemodel

python demo_image_det_ccpd.py --model ../prototxt/deep_ccpd_det_deploy_v1.prototxt --weights ../snapshot/deepccpd_res_v1_iter_370000.caffemodel --ccpdmodel ../prototxt/deep_ccpd_rec_deploy_v2.prototxt --ccpdweights ../snapshot/deepccpd_rec_v2_iter_775000.caffemodel
