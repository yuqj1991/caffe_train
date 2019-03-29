#!/usr/bin/env python

from tool_csv import loadCSVFile
from tool_lxml import createXML
import cv2
import os

FILEDIR = "/media/scs4450/hard/umdfaces_batch1/"         #
IMGSTORE = "/media/scs4450/hard/JPEGImages/"
FILENAME = "umdfaces_batch1_ultraface.csv"               #
ANNOTATIONDIR = "/media/scs4450/hard/Annotations/"

if __name__ == "__main__":
    csv_content = loadCSVFile(FILEDIR+FILENAME)
    cvs_content_part = csv_content[1:,1:10]
    i=1
    base=3000000                                         #
    limit = 1000000                                       #
    
    for info in cvs_content_part:
        if i==limit:
            print "Reach Limit, Stop..."
            break

        print "Process No." + str(i) + " Data...."

        str_splite = '/'
        str_spilte_list = str(info[0]).split(str_splite)

        jpg_path = info[0]
        #jpg_file = str_spilte_list[len(str_spilte_list)-1]
        jpg_file = str(base+i)+'.jpg'
        os.system('cp '+ FILEDIR+jpg_path + ' ' + IMGSTORE+jpg_file)

        img = cv2.imread(FILEDIR+jpg_path)
        sp = img.shape
        #print sp
        height = sp[0]                 #height(rows) of image
        width = sp[1]                  #width(colums) of image
        depth = sp[2]                  #the pixels value is made up of three primary colors
        #print 'width: %d \nheight: %d \nnumber: %d' %(width,height,depth)

        xmin = int(float(info[3]))
        ymin = int(float(info[4]))
        xmax = int(float(info[3])+float(info[5]))
        ymax = int(float(info[4])+float(info[6]))
        #print 'xmin: %d \nymin: %d \nxmax: %d \nymax: %d' %(xmin,ymin,xmax,ymax)

        transf = dict()
        transf['folder'] = "FACE2016"
        transf['filename'] = jpg_file
        transf['width'] = str(width)
        transf['height'] = str(height)
        transf['depth'] = str(depth)
        transf['xmin'] = str(xmin)
        transf['ymin'] = str(ymin)
        transf['xmax'] = str(xmax)
        transf['ymax'] = str(ymax)

        print "Create No." + str(i) + " XML...."
        createXML(transf,ANNOTATIONDIR)
        i = i + 1
        #print jpg_path, jpg_file
        #jpg
    
    print "Done..."

    