import numpy as np

import sys

caffe_root = '../../../../caffe_train/'
sys.path.insert(0, caffe_root + 'python')
import caffe
blob = caffe.proto.caffe_pb2.BlobProto()

'''
with open('mean.npy','rb') as f:
     data_mean = numpy.load(f)
'''


'''
print blob.channels
print "val is ",float(sys.argv[1])
Tem = Temp_3A[0]*(float(sys.argv[1]))
print "Tem",Tem
#get command line fun
'''

#print "cha ",cha_1_val

#define diff channel's val
#Temp_2B = np.ones(256,256)
def get_argv():
		cha_1_val = float(sys.argv[1])
		cha_2_val = float(sys.argv[2])
		cha_3_val = float(sys.argv[3])
		blob_height = int(sys.argv[4])
		blob_width = int(sys.argv[5])
		return cha_1_val,cha_2_val,cha_3_val,blob_height,blob_width

def make_bin(cha_1_val,cha_2_val,cha_3_val,blob_height,blob_width):
		#define a 3 dim array
		Temp_3A = np.ones([3,blob_height,blob_width],dtype = np.float)
		blob.num = 1
		blob.channels,blob.height,blob.width = Temp_3A.shape		

		Temp_3A[0] = np.ones([blob_height,blob_width])*cha_1_val
		Temp_3A[1] = np.ones([blob_height,blob_width])*cha_2_val
		Temp_3A[2] = np.ones([blob_height,blob_width])*cha_3_val

		#print "Temp is ",Temp_3A

		blob.data.extend(Temp_3A.astype(float).flat)
		#print "blob data",blob

		binaryproto_file = open('mean.binaryproto','wb')
		binaryproto_file.write(blob.SerializeToString())
		binaryproto_file.close()
		#print "make file success!"
if __name__ == "__main__":
		#print len(sys.argv)
		if len(sys.argv) != 6:
				print ("Usage:python **.py [B] [G] [R] [image_height] [image_width]")
		else:
				cha_1_val,cha_2_val,cha_3_val,blob_height,blob_width = get_argv()
				#print "make binaryproto file..."
                                #print "B:%f,G:%f,R:%f,img_height:%d,img_width:%d"%(cha_1_val,cha_2_val,cha_3_val,blob_height,blob_width)
				make_bin(cha_1_val,cha_2_val,cha_3_val,blob_height,blob_width)

