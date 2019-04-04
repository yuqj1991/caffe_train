import cv2
import common
import datetime
import os

SOURCE_IMG_FILE_FOLDER = '../../../dataset/facedata/umdface/JPEGImages/' 
LABEL_FILE_FOLDER = '../../../dataset/facedata/umdface/labels/'
CSV_FILE_NAME = 'roi.csv'

def batch_work():
    ori = ((common.ORI_BATCH1, common.PROCESSED_BATCH1), (common.ORI_BATCH2, common.PROCESSED_BATCH2), (common.ORI_BATCH3, common.PROCESSED_BATCH3))
    for pairs in ori:
        df = common.read_from_file(pairs[0]+CSV_FILE_NAME)
        for row in df.iterrows():
            #Extract Important Imformation
            file_name = row[1]['FILE']
            img_file_name_no_jpg = file_name.split('/')[1].split('.jpg')[0]
            label_full_anno_file_name = LABEL_FILE_FOLDER + img_file_name_no_jpg + '.txt'
            label_angle_anno_file_name = LABEL_FILE_FOLDER + img_file_name_no_jpg + '_angle.txt'
            full_path_image_name = SOURCE_IMG_FILE_FOLDER + pairs[0] + file_name
            print('label file: %s, and full_path_img : %s'%(label_full_anno_file_name, full_path_image_name))
            label_file_ = open(label_full_anno_file_name, 'w')
            label_file_angle = open(label_angle_anno_file_name, 'w')
            roi_x = int(row[1]['FACE_X'])
            roi_y = int(row[1]['FACE_Y'])
            roi_w = int(row[1]['FACE_WIDTH'])
            roi_h = int(row[1]['FACE_HEIGHT'])
            ponit_x1 = row[1]['P1X']
            ponit_x2 = row[1]['P2X']
            ponit_x3 = row[1]['P3X']
            ponit_x4 = row[1]['P4X']
            ponit_x5 = row[1]['P5X']
            ponit_x6 = row[1]['P6X']
            ponit_x7 = row[1]['P7X']
            ponit_x8 = row[1]['P8X']
            ponit_x9 = row[1]['P9X']
            ponit_x10 = row[1]['P10X']
            ponit_x11 = row[1]['P11X']
            ponit_x12 = row[1]['P12X']
            ponit_x13 = row[1]['P13X']
            ponit_x14 = row[1]['P14X']
            ponit_x15 = row[1]['P15X']
            ponit_x16 = row[1]['P16X']
            ponit_x17 = row[1]['P17X']
            ponit_x18 = row[1]['P18X']
            ponit_x19 = row[1]['P19X']
            ponit_x20 = row[1]['P20X']
            ponit_x21 = row[1]['P21X']
            ponit_y1 = row[1]['P1Y']
            ponit_y2 = row[1]['P2Y']
            ponit_y3 = row[1]['P3Y']
            ponit_y4 = row[1]['P4Y']
            ponit_y5 = row[1]['P5Y']
            ponit_y6 = row[1]['P6Y']
            ponit_y7 = row[1]['P7Y']
            ponit_y8 = row[1]['P8Y']
            ponit_y9 = row[1]['P9Y']
            ponit_y10 = row[1]['P10Y']
            ponit_y11 = row[1]['P11Y']
            ponit_y12 = row[1]['P12Y']
            ponit_y13 = row[1]['P13Y']
            ponit_y14 = row[1]['P14Y']
            ponit_y15 = row[1]['P15Y']
            ponit_y16 = row[1]['P16Y']
            ponit_y17 = row[1]['P17Y']
            ponit_y18 = row[1]['P18Y']
            ponit_y19 = row[1]['P19Y']
            ponit_y20 = row[1]['P20Y']
            ponit_y20 = row[1]['P20Y']
            ponit_y21 = row[1]['P21Y']
            yaw = row[1]['YAW']
            pitch = row[1]['PITCH']
            roll = row[1]['ROLL']
            content = ponit_x1 + ' ' + ponit_x2 + ' ' + ponit_x3 + ' ' + ponit_x4 + ' ' + ponit_x5 + ' ' + ponit_x6 + ' ' + ponit_x7 + ' ' 
                        + ponit_x8 + ' ' + ponit_x9 + ' ' + ponit_x10 + ' ' + ponit_x11 + ' ' + ponit_x12 + ' ' + ponit_x13 + ' ' + ponit_x14 + ' '
                        + ponit_x15 + ' ' + ponit_x16 + ' ' + ponit_x17 + ' ' + ponit_x18 + ' ' + ponit_x19 + ' ' + ponit_x20 + ' ' + ponit_x21 + ' '
                        + ponit_y1 + ' ' + ponit_y2 + ' ' + ponit_y3 + ' ' + ponit_y4 + ' ' + ponit_y5 + ' ' + ponit_y6 + ' ' + ponit_y7 + ' ' 
                        + ponit_y8 + ' ' + ponit_y9 + ' ' + ponit_y10 + ' ' + ponit_y11 + ' ' + ponit_y12 + ' ' + ponit_y13 + ' ' + ponit_y14 + ' ' 
                        + ponit_y15 + ' ' + ponit_y16 + ' ' + ponit_y17 + ' '+ ponit_y18 + ' ' + ponit_y19 + ' ' + ponit_y20 + ' ' + ponit_y21 + ' '
                        + yaw + ' ' + pitch + ' ' + roll + '\n'
            label_file_.write(content)
            label_file_angle.write(yaw + ' ' + pitch + ' ' + roll + '\n')
            label_file_angle.close()
            label_file_.close()

if __name__ == '__main__':
    LOG = common.init_my_logger()
    start_time = datetime.datetime.now()
    batch_work()
    end_time = datetime.datetime.now()
    LOG.info('Cost %d Seconds!'%((end_time-start_time).seconds))
