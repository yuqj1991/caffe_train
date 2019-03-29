import cv2
import common
import datetime
import os

CSV_FILE_NAME = 'roi.csv'

def batch_work():
    ori = ((common.ORI_BATCH1, common.PROCESSED_BATCH1), (common.ORI_BATCH2, common.PROCESSED_BATCH2), (common.ORI_BATCH3, common.PROCESSED_BATCH3))
    for pairs in ori:
        df = common.read_from_file(pairs[0]+CSV_FILE_NAME)
        for row in df.iterrows():
            #Extract Important Imformation
            file_name = row[1]['FILE']
            roi_x = int(row[1]['FACE_X'])
            roi_y = int(row[1]['FACE_Y'])
            roi_w = int(row[1]['FACE_WIDTH'])
            roi_h = int(row[1]['FACE_HEIGHT'])
            #Create Dir if not Exist
            file_name = file_name.strip()
            dir_name = file_name.split('/')[0]
            if not os.path.isdir(pairs[1]+dir_name):
                os.makedirs(pairs[1]+dir_name)
            #Crop And Resize Image
            img = cv2.imread(pairs[0]+file_name)
            img_roi = img[roi_y:(roi_y+roi_h+1), roi_x:(roi_x+roi_w+1)]
            img_roi_resize = cv2.resize(img_roi, (182, 182))
            cv2.imwrite(pairs[1]+file_name, img_roi_resize)
            LOG.info('Process %s Done!'%(file_name))

if __name__ == '__main__':
    LOG = common.init_my_logger()
    start_time = datetime.datetime.now()
    batch_work()
    end_time = datetime.datetime.now()
    LOG.info('Cost %d Seconds!'%((end_time-start_time).seconds))