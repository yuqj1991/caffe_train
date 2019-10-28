import numpy as np
import h5py
import os
import cv2
import random
import sys
import random
from scipy import misc

def prepare_data(path):
    f = h5py.File('%s/cuhk-03.mat' % path)
    labeled = [f['labeled'][0][i] for i in range(len(f['labeled'][0]))]
    labeled = [f[labeled[0]][i] for i in range(len(f[labeled[0]]))]
    detected = [f['detected'][0][i] for i in range(len(f['detected'][0]))]
    detected = [f[detected[0]][i] for i in range(len(f[detected[0]]))]
    datasets = [['labeled', labeled], ['detected', detected]]
    prev_id = 0

    for dataset in datasets:
        if not os.path.exists('%s/%s/train' % (path, dataset[0])):
            os.makedirs('%s/%s/train' % (path, dataset[0]))
        if not os.path.exists('%s/%s/val' % (path, dataset[0])):
            os.makedirs('%s/%s/val' % (path, dataset[0]))

        for i in range(0, len(dataset[1])):
            for j in range(len(dataset[1][0])):
                try:
                    image = np.array(f[dataset[1][i][j]]).transpose((2, 1, 0))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = cv2.imencode('.jpg', image)[1].tostring()
                    if len(dataset[1][0]) - j <= 100:
                        filepath = '%s/%s/val/%04d_%02d.jpg' % (path, dataset[0], j - prev_id - 1, i)
                    else:
                        filepath = '%s/%s/train/%04d_%02d.jpg' % (path, dataset[0], j, i)
                        prev_id = j
                    with open(filepath, 'wb') as image_file:
                        image_file.write(image)
                except Exception as e:
                    continue

def get_pair(path, set, num_id, positive):
    pair = []
    if positive:
        value = int(random.random() * num_id)
        id = [value, value]
    else:
        while True:
            id = [int(random.random() * num_id), int(random.random() * num_id)]
            if id[0] != id[1]:
                break

    for i in range(2):
        filepath = ''
        while True:
            index = int(random.random() * 10)
            filepath = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id[i], index)
            if not os.path.exists(filepath):
                continue
            break
        pair.append(filepath)
    return pair

def get_num_id(path, set):
    files = os.listdir('%s/labeled/%s' % (path, set))
    files.sort()
    return int(files[-1].split('_')[0]) - int(files[0].split('_')[0]) + 1

def read_data(path, set, num_id, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    for i in range(batch_size // 2):
        pairs = [get_pair(path, set, num_id, True), get_pair(path, set, num_id, False)]
        for pair in pairs:
            images = []
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            batch_images.append(images)
        labels.append([1., 0.])
        labels.append([0., 1.])

    '''
    for pair in batch_images:
        for p in pair:
            cv2.imshow('img', p)
            key = cv2.waitKey(0)
            if key == 1048603:
                exit()
    '''
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)


def getTrainData(imgSetfile, netInputWidth, netInputHeight): #imageSetfile[imagepath, label, ...]
    batch_images = []
    labels = []
    for imgfile in imgSetfile:
        imgPath = imgfile.split(' ')[0]
        label = int(imgfile.split(' ')[1])
        imageData = cv2.imread(imgPath)
        imageData = cv2.resize(imageData, (netInputWidth, netInputHeight))
        imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB)
        batch_images.append(imageData)
        labels.append(label)


def random_rotate_image(image):
    random_angle = np.random.uniform(low = -10.0, hight = 10.0)
    return misc.imrotate(image, angle= random_angle)


def random_crop_image(image, crop_shape):
    assert image.shape[0] >= crop_shape[0]
    assert image.shape[1] >= crop_shape[1]

    offset_x = random.randint(0, image.shape[1] - crop_shape[1])
    offset_y = random.randint(0, image.shape[0] - crop_shape[0])

    cropImage = image[offset_y:offset_y + crop_shape[1], offset_x + crop_shape[0]]
    return cropImage


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image






if __name__ == '__main__':
    prepare_data("../../../../../dataset/reId_data/cuhk03_release")
