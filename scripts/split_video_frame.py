import numpy as np
import cv2
import os

SRC_VIDEOS_DIR = '/home/deepano/workspace/dataset/roadSign/video'
DIST_FRAME_DIR = '/home/deepano/workspace/dataset/roadSign/frame'

def get_frame(src_video_dir, dist_frame_dir, fps):
	for _file in os.listdir(src_video_dir):
		video_file = os.path.join(src_video_dir, _file)
		print(video_file)
		cap=cv2.VideoCapture(video_file)
		count = 0
		while (True):
			ret,frame=cap.read()
			if ret == True:
				if not os.path.isdir(dist_frame_dir):  # Create the log directory if it doesn't exist
					os.makedirs(dist_frame_dir)
				#if count % 
				frame_name = _file.split('.')[0] + '_' + str(count) + '.jpg'
				frame_file = os.path.join(dist_frame_dir, frame_name)
				cv2.imwrite(frame_file, frame)
				count += 1
			else:
				break
		cap.release()
		

def main():
	get_frame(SRC_VIDEOS_DIR, DIST_FRAME_DIR, 0)
	
if __name__=="__main__":
	main()
