import os
import cv2
import time
import numpy as np
from utils import util_data
import tensorflow as tf
model_checkpoint = "/home/resideo/workspace/action_recognition/resideo_action/models/action/20190428-165230"
classInd_file = "/home/resideo/workspace/dataset/actiondata/ucf101/train_test_indices/classInd.txt"


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


def prewhiten(x):
    y = np.multiply(np.subtract(x, 127.5), 1/127.5)
    return y


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            util_data.load_model(model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        batchsize_placeholder = tf.get_default_graph().get_tensor_by_name("batch_size:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("result/prediction:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        prewhiten_face = prewhiten(face)
        batch_size = 1
        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False,
                     batchsize_placeholder: batch_size}
        return self.sess.run(embeddings, feed_dict=feed_dict)[-1]


def main():
    action_ = Encoder()
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()
    frame_count = 0
    frame_interval = 3
    fps_display_interval = 5
    class_action = []
    with open(classInd_file, 'r') as lab_file:
        lab_an_line = lab_file.readline()
        while lab_an_line:
            class_action.append(lab_an_line)
            lab_an_line = lab_file.readline()
        lab_file.close()
    while True:
        # Capture frame-by-frame
        ret, frame_ = video_capture.read()
        frame = cv2.resize(frame_, (299, 299), interpolation=cv2.INTER_AREA)
        if (frame_count % frame_interval) == 0:
            embedding = action_.generate_embedding(frame)
            embedding_ = np.array(embedding)
            embedding_max = np.argmax(embedding_)
            result_content = "the result is: " + class_action[embedding_max] + ", and the confidence score is: "+ \
                                str(embedding_[embedding_max])
            print(result_content)
            cv2.putText(frame, result_content,  (10, 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),
                thickness=1, lineType=1)
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
        frame_count += 1
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

