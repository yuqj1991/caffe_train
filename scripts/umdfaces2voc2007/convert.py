import cv2
import common
import datetime
import os

SOURCE_IMG_FILE_FOLDER = '../../../dataset/facedata/umdface/JPEGImages/' 
LABEL_FILE_FOLDER = '../../../dataset/facedata/umdface/labels/'
ImageSetFileForder = '../../../dataset/facedata/umdface/ImageSet/Main/'
trainFileName = ['umdfaces_batch1_ultraface.csv', 'umdfaces_batch2_ultraface.csv']
testFileName = ['umdfaces_batch3_ultraface.csv']
trainSet = [common.ORI_BATCH1, common.ORI_BATCH2]
testSet = [ common.ORI_BATCH3]

def batch_work(ori, csvFile, setFile):
    setfile_ = open(setFile, 'w')
    for ii in range(len(ori)):
        df = common.read_from_file(SOURCE_IMG_FILE_FOLDER+ori[ii]+csvFile[ii])
        for row in df.iterrows():
            #Extract Important Imformation
            file_name = row[1]['FILE']
            img_file_name_no_jpg = file_name.split('/')[1].split('.jpg')[0]
            label_full_anno_file_name = LABEL_FILE_FOLDER + img_file_name_no_jpg + '.txt'
            label_angle_anno_file_name = LABEL_FILE_FOLDER + img_file_name_no_jpg + '_angle.txt'
            full_path_image_name = SOURCE_IMG_FILE_FOLDER + ori[ii] + file_name
            fullImg = os.path.abspath(full_path_image_name) + '\n'
            setfile_.writelines(fullImg)
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
            content = str(ponit_x1) + ' ' + str(ponit_x2) + ' ' + str(ponit_x3) + ' ' + str(ponit_x4) + ' ' + str(ponit_x5) + ' ' + str(ponit_x6) + ' ' + str(ponit_x7) + ' ' + str(ponit_x8) + ' ' + str(ponit_x9) + ' ' + str(ponit_x10) + ' ' + str(ponit_x11) + ' ' + str(ponit_x12) + ' ' + str(ponit_x13) + ' ' + str(ponit_x14) + ' '+ str(ponit_x15) + ' ' + str(ponit_x16) + ' ' + str(ponit_x17) + ' ' + str(ponit_x18) + ' ' + str(ponit_x19) + ' ' + str(ponit_x20) + ' ' + str(ponit_x21) + ' '+ str(ponit_y1) + ' ' + str(ponit_y2) + ' ' + str(ponit_y3) + ' ' + str(ponit_y4) + ' ' + str(ponit_y5) + ' ' + str(ponit_y6) + ' ' + str(ponit_y7) + ' ' + str(ponit_y8) + ' ' + str(ponit_y9) + ' ' + str(ponit_y10) + ' ' + str(ponit_y11) + ' ' + str(ponit_y12) + ' ' + str(ponit_y13) + ' ' + str(ponit_y14) + ' ' + str(ponit_y15) + ' ' + str(ponit_y16) + ' ' + str(ponit_y17) + ' '+ str(ponit_y18) + ' ' + str(ponit_y19) + ' ' + str(ponit_y20) + ' ' + str(ponit_y21) + ' '+ str(yaw) + ' ' + str(pitch) + ' ' + str(roll) + '\n'
            label_file_.write(content)
            label_file_angle.write(str(yaw) + ' ' + str(pitch) + ' ' + str(roll) + '\n')
            label_file_angle.close()
            label_file_.close()
    setfile_.close()
   

if __name__ == '__main__':
    LOG = common.init_my_logger()
    start_time = datetime.datetime.now()
    trainset = ImageSetFileForder + 'train.txt'
    testset = ImageSetFileForder + 'test.txt'
    batch_work(trainSet, trainFileName, trainset)
    batch_work(testSet, testFileName, testset)
    end_time = datetime.datetime.now()
    LOG.info('Cost %d Seconds!'%((end_time-start_time).seconds))
