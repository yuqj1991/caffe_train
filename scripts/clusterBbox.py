# -*- coding:UTF-8 -*-
from __future__ import division
import numpy as np


classfyFile = "./roadSign_classfly_distance_data.txt"
#classfyFile = './ccpd_classfly_distance_data.txt'

# 定义Box类，描述bounding box的坐标
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
 
 
# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2
    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)
    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area
 
 
# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)
    

def avg_iou(boxes, centroids):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    sum_distance = 0
    for box in boxes:
        one_distance = []
        for centroid_index, centroid in enumerate(centroids):
            if 0:
                print("box.x %f, box.y: %f, box.w: %f, box.h:%f"%(box.x,box.y, box.w, box.h))
                print("centroid.x %f, centroid.y: %f, centroid.w: %f, centroid.h:%f"%(centroid.x,centroid.y, centroid.w, centroid.h))
                print('box_iou(box, centroid): %f'%(box_iou(box, centroid)))
            distance = (1 - box_iou(box, centroid))
            one_distance.append(distance)
        sum_distance += max(one_distance)
    return sum_distance / len(boxes)


def bboxIou(bbox_one, bbox_two):
	center_x_one = bbox_one[0]
	center_y_one = bbox_one[1]
	bbox_one_width = bbox_one[2]
	bbox_one_height = bbox_one[3]
	center_x_two = bbox_two[0]
	center_y_two = bbox_two[1]
	bbox_two_width = bbox_two[2]
	bbox_two_height = bbox_two[3]
	if (center_x_two-bbox_two_width/2) <(center_x_one+bbox_one_width/2) and (center_y_two-bbox_two_height/2)<(center_y_one+bbox_one_height/2):
		jessord_area  = (center_x_one+bbox_one_width/2 - (center_x_two-bbox_two_width/2))*(center_y_one+bbox_one_height/2 - (center_y_two-bbox_two_height/2))
		total_area = bbox_one_width* bbox_one_height+ bbox_two_width* bbox_two_height-jessord_area
		return float(jessord_area/total_area)
	else:
		return 0.0


def getClassflyIouBbox(annoBboxDatafile):
	bboxlist = []
	with open(annoBboxDatafile, 'r') as file_:
		while True:
			lineinfo = file_.readline().replace('\n', '').split(' ')
			if lineinfo[0]=='':
				break
			center_x = float(lineinfo[0])
			center_y = float(lineinfo[1])
			class_width = float(lineinfo[2])
			class_height = float(lineinfo[3])
			bboxlist.append(Box(0, 0, class_width, class_height))
		file_.close()
	return bboxlist


# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)
 
    centroid_index = np.random.choice(boxes_num, 1)
    centroids.append(boxes[centroid_index[0]])
 
    print(centroids[0].w,centroids[0].h)
 
    for centroid_index in range(0,n_anchors-1):
 
        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0
 
        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)
 
        distance_thresh = sum_distance*np.random.random()
 
        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break
 
    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    distance_all = []
    eps = 0.005
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))
 
    for box in boxes:
        min_distance = 1
        group_index = 0
        one_distance = []
        for centroid_index, centroid in enumerate(centroids):
            if 0:
                print("box.x %f, box.y: %f, box.w: %f, box.h:%f"%(box.x,box.y, box.w, box.h))
                print("centroid.x %f, centroid.y: %f, centroid.w: %f, centroid.h:%f"%(centroid.x,centroid.y, centroid.w, centroid.h))
                print('box_iou(box, centroid): %f'%(box_iou(box, centroid)))
            distance = (1 - box_iou(box, centroid))
            one_distance.append(distance)
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        distance_all.append(one_distance)
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h
    for i in range(n_anchors):
		new_centroids[i].w /= (len(groups[i])+eps)
		new_centroids[i].h /= (len(groups[i])+eps)
    return new_centroids, groups, loss, distance_all


# 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集bbox anno setfile
# n_anchors 是anchors的数量
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def compute_centroids(label_path,n_anchors,loss_convergence,grid_size,iterations_num,plus):
    boxes = getClassflyIouBbox(label_path)
    if plus:
		print('boxes length: ', len(boxes))
		centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])
 
    # iterate k-means
    centroids, groups, old_loss, distance_all_old = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    prev_assignments = np.ones(len(boxes))*(-1)    
    while (True):
        centroids, groups, loss, distance_all = do_kmeans(n_anchors, boxes, centroids)
        assignments = np.argmin(distance_all, axis=1)
        iterations = iterations + 1
        print("~~~~~~~~~~~~~~~the %d times iterations~~~~~~~~~~~~~~~~~~~~~~~"%(iterations+1))
        print("old_loss = %f, loss = %f" % (old_loss, loss))
        if abs(old_loss-loss) < loss_convergence or iterations > iterations_num:
        #if (assignments == prev_assignments).all() :
            break
        old_loss = loss 
        prev_assignments = assignments.copy()
    # print result
    ii=0
    for centroid in centroids:
        print("k-means result：\n")
        print("the num of the group_%d: %d"%(ii, len(groups[ii])))
        print(centroid.w*grid_size, centroid.h*grid_size)
        ii+=1
    avgIOU = avg_iou(boxes, centroids)
    print("avgIOU: ", avgIOU)


def main():
	if 1:
		n_anchors = 9
		loss_convergence = 1e-2
		grid_size = 416
		iterations_num = 10000
		plus = 1
		compute_centroids(classfyFile,n_anchors,loss_convergence,grid_size,iterations_num,plus)


if __name__ == '__main__':
	main()


