import shutil
import os

market_1501_dir = '../../dataset/reId_data/Market-1501-v15.09.15/gt_bbox'
mark_1501_align_dir = '../../dataset/reId_data/Market-1501_align'

def generate_name_dir(srcDir, disDir):
    if not os.path.isdir(disDir):
        os.makedirs(disDir)
    for imgfile in os.listdir(srcDir):
        srcimgfile = srcDir + '/' + imgfile
        class_name = 'n' + imgfile.split('_')[0]
        classDir = disDir + "/" + class_name
        distimgfile = classDir + '/' + imgfile
        if not os.path.isdir(classDir):
            os.makedirs(classDir)
            shutil.copyfile(srcimgfile, distimgfile)
        else:
            for class_id in os.listdir(disDir):
                if class_name == class_id:
                    shutil.copyfile(srcimgfile, distimgfile)

def main():
    generate_name_dir(market_1501_dir, mark_1501_align_dir)

if __name__=='__main__':
    main()
