import cv2
import common
import datetime
import os
import numpy as np

SOURCE_IMG_FILE_FOLDER = '../../../dataset/facedata/umdface/JPEGImages/'
CROP_IMG_FILE_FOLDER = '../../../dataset/facedata/umdface/annoImages/'
CROP_IMG_FILE_FOLDER_DRAW = '../../../dataset/facedata/umdface/cropImg/'
LABEL_FILE_FOLDER = '../../../dataset/facedata/umdface/labels/'
ImageSetFileForder = '../../../dataset/facedata/umdface/ImageSet/Main/'
annoDict = {common.ORI_BATCH1: 'umdfaces_batch1_ultraface_new.csv', common.ORI_BATCH2: 'umdfaces_batch2_ultraface_new.csv', common.ORI_BATCH3: 'umdfaces_batch3_ultraface_new.csv'}
trainDataSet = [common.ORI_BATCH1, common.ORI_BATCH2]
testDataSet = [common.ORI_BATCH3]
maxMargin = 66
minMargin = 36
def batch_work(ori, setFile):
    setfile_ = open(setFile, 'w')
    for annofile in ori:
        df = common.read_from_file(SOURCE_IMG_FILE_FOLDER+ annofile + annoDict[annofile])
        for row in df.iterrows():
            #Extract Important Imformation
            file_name = row[1]['FILE']
            img_file_name_no_jpg = file_name.split('/')[1].split('.jpg')[0]
            label_full_anno_file_name = LABEL_FILE_FOLDER + img_file_name_no_jpg + '.txt'
            full_path_image_name = SOURCE_IMG_FILE_FOLDER + annofile + file_name
            if not os.path.exists(CROP_IMG_FILE_FOLDER + annofile + file_name.split('/')[0]):
                os.mkdir(CROP_IMG_FILE_FOLDER + annofile + file_name.split('/')[0])
            ang_path_image_name = CROP_IMG_FILE_FOLDER + annofile + file_name.split('.jpg')[0]+'_crop.jpg'
            fullImg = os.path.abspath(full_path_image_name) + '\n'
            print('label file: %s, and full_path_img : %s'%(label_full_anno_file_name, full_path_image_name))
            label_file_ = open(label_full_anno_file_name, 'w')
            roi_x = int(row[1]['FACE_X'])
            roi_y = int(row[1]['FACE_Y'])
            roi_w = int(row[1]['FACE_WIDTH'])
            roi_h = int(row[1]['FACE_HEIGHT'])
            yaw = row[1]['YAW']
            pitch = row[1]['PITCH']
            roll = row[1]['ROLL']
            pr_female = row[1]['PR_FEMALE']
            pr_male = row[1]['PR_MALE']
            boolGlass = row[1]['BOOLGLASS']
            src = cv2.imread(os.path.abspath(full_path_image_name))
            xmin = np.maximum(roi_x - minMargin / 2, 0)
            xmax = np.minimum(roi_x + roi_w + minMargin / 2, src.shape[1])
            ymin = np.maximum(roi_y - maxMargin / 2, 0)
            ymax = np.minimum(roi_y+roi_h + minMargin / 2, src.shape[0])
            cropRoi = src[ymin:ymax, xmin:xmax, :]
            left_eye_point_x = row[1]['P8X'] - xmin
            right_eye_point_x = row[1]['P11X'] -xmin
            nose_point_x = row[1]['P15X'] -xmin
            left_mouse_point_x = row[1]['P18X'] -xmin
            right_mouse_point_x = row[1]['P20X'] - xmin
            left_eye_point_y = row[1]['P8Y'] - ymin
            right_eye_point_y = row[1]['P11Y'] -ymin
            nose_point_y = row[1]['P15Y'] - ymin
            left_mouse_point_y = row[1]['P18Y'] -ymin
            right_mouse_point_y = row[1]['P20Y'] - ymin
            cv2.imwrite(ang_path_image_name, cropRoi)
            if 1:
                pointSet = []
                pointSet.append((int(left_eye_point_x), int(left_eye_point_y)))
                pointSet.append((int(right_eye_point_x), int(right_eye_point_y)))
                pointSet.append((int(nose_point_x), int(nose_point_y)))
                pointSet.append((int(left_mouse_point_x), int(left_mouse_point_y)))
                pointSet.append((int(right_mouse_point_x), int(right_mouse_point_y)))
                for ii in range(5):
                    cv2.circle(cropRoi, pointSet[ii], 3, (0,0,213), -1)
                ang_path_image_name_new = CROP_IMG_FILE_FOLDER_DRAW + img_file_name_no_jpg + '_crop.jpg'
                cv2.imwrite(ang_path_image_name_new, cropRoi)

            setfile_.writelines(os.path.abspath(ang_path_image_name) + '\n')
            content = str(left_eye_point_x) + ' ' + str(right_eye_point_x) + ' ' + str(nose_point_x) + ' ' + str(left_mouse_point_x) + ' ' + str(right_mouse_point_x) + ' ' + str(left_eye_point_y) + ' ' + str(right_eye_point_y) + ' ' + str(nose_point_y) + ' ' + str(left_mouse_point_y) + ' ' + str(right_mouse_point_y) + ' ' + str(yaw) + ' ' + str(pitch) + ' ' + str(roll) + ' ' + str(pr_female) + ' ' + str(pr_male) + ' ' + str(boolGlass) + '\n'
            label_file_.write(content)
            label_file_.close()
    setfile_.close()


if __name__ == '__main__':
    LOG = common.init_my_logger()
    start_time = datetime.datetime.now()
    trainset = ImageSetFileForder + 'training_umdface_pose.txt'
    testset = ImageSetFileForder + 'testing_umdface_pose.txt'
    batch_work(trainDataSet, trainset)
    batch_work(testDataSet, testset)
    end_time = datetime.datetime.now()
    LOG.info('Cost %d Seconds!'%((end_time-start_time).seconds))
