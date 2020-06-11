import sys, os
import numpy as np
import json
from coco.PythonAPI.pycocotools import COCO
import tqdm

numberKeyPoints = 17


def getFilesInDirectry(fileDir, fileExtension):
    filePaths = []
    if not os.path.isfile(fileDir) and os.path.isdir(fileDir):
        for subfile in os.listdir(fileDir):
            if isinstance(fileExtension, str):
                ext = subfile.split('.')[-1]
                if(fileExtension == ext):
                    filePaths.append(subfile)
    return filePaths


def get_json_negatives_sample():
    '''
    get no person images id, this is test code!
    '''
    annoTypes = ['instances', 'captions', 'person_keypoints']
    annoType = annoTypes[2]
    dataType = "train2017"
    imageFilePaths = getFilesInDirectry("folder", "jpg")
    numberImages = len(imageFilePaths)
    imageIds = []
    for i in range(numberImages):
        imageIds.append(int(imageFilePaths[i].split('_')[-1].split('.')[0]))
    print("Reading JSON people annotations...\n")
    annotationsFile = "{}_{}.json".format(annoType, dataType)
    coco = COCO(annotationsFile)
    jsonPeopleAnnotations = coco.dataset['annotations']
    numberAnnotations = len(jsonPeopleAnnotations)
    imagePeopleIds = []
    for i in range(numberAnnotations):
        imagePeopleIds.append(jsonPeopleAnnotations[i]['image_id'])
    imagePeopleIds = np.array(imagePeopleIds) 
    imagePeopleIds = np.unique(imagePeopleIds)
    numberAnnotations = len(imagePeopleIds)
    imageNopeopleIds = list(set(imageIds), set(imagePeopleIds))
    numberImagesNoPeople = len(imageNopeopleIds)
    imageWithNoPeoplePaths = []
    for i in range(numberImagesNoPeople):
        imageWithNoPeoplePaths.append(imageNopeopleIds[i])
    return imageWithNoPeoplePaths


def get_json_annotations_to_voc():
    '''
    get annotations from coco jsons
    '''
    


def get_json_mask_to_voc():
    '''
    get json mask to voc
    '''


def refineAnnotations():
    '''
    refined annotations from json
    '''



