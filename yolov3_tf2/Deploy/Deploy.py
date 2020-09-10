import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import flask
from flask import Flask,request
from collections import OrderedDict
import io
import base64
import numpy
from imageio import imread
classes='./data/coco.names'
weights='./checkpoints/yolov3.tf'
tiny=False,
size= 416
image= './data/girl.png'
tfrecord= None
output='./output.jpg'
num_classes= 80
class_names=[]
yolo=None

def initializations():
    global yolo
    global weights
    global classes
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=classes)
    else:
        yolo = YoloV3(classes=classes)

    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')


initializations()
app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def main():
    sbuf = io.StringIO()
    _json_response = OrderedDict()
    global size,class_names,yolo
    if flask.request.method == "POST":
        if request.data:
            max_class_name = 'No detection'
            second_max_class_name = 'No detection'
            max_score = 0
            second_max_score = 0
            max_result_roi = None
            second_max_roi = None
            max_result_roi_string = 'None'
            second_max_roi_string = 'None'
            # print(type(request.data))
            b64_string = request.data.decode()
            print("Size of base64 recieved")
            print(len(b64_string))
            # sbuf.write(base64.b64decode(request.data.decode()))
            # nparr = np.fromstring(request.data, np.uint8)
            bytes_image = io.BytesIO(base64.b64decode(b64_string))
            image1 = imread(bytes_image)
            np_im = np.array(image1)
            image_predict=tf.image.decode_image(np_im, channels=3)
            img = tf.expand_dims(image_predict, 0)
            img = transform_images(img,size)
            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)

            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                   np.array(scores[0][i]),
                                                   np.array(boxes[0][i])))
    # if FLAGS.tfrecord:
    #     dataset = load_tfrecord_dataset(
    #         FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
    #     dataset = dataset.shuffle(512)
    #     img_raw, _label = next(iter(dataset.take(1)))
    # else:
    #     img_raw = tf.image.decode_image(
    #         open(FLAGS.image, 'rb').read(), channels=3)
    #
    # img = tf.expand_dims(img_raw, 0)
    # img = transform_images(img, FLAGS.size)


if __name__ == '__main__':
    app.run(host=0.0.0.0)
