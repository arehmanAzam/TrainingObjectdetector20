import time
from absl import app, logging
# from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from TrainingObjectdetector20.yolov3_tf2.yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from TrainingObjectdetector20.yolov3_tf2.yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from TrainingObjectdetector20.yolov3_tf2.yolov3_tf2.utils import draw_outputs
import flask
from flask import Flask,request
from collections import OrderedDict
import io
import base64
import numpy
from imageio import imread
import os
import json

classes='../data/voc2012.names'
weights='../checkpoints/yolov3_train_11.tf'
tiny=False,
size= 416
image= './data/girl.png'
tfrecord= None
output='./output.jpg'
num_classes= 2
class_names=[]
yolo=None
first_run_flag=True

flags = tf.compat.v1.flags

def initializations():
    global yolo
    global weights
    global classes
    global class_names,num_classes
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=num_classes)

    checkpoint_dir = os.path.dirname(weights)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    logging.info("loading model")
    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')



initializations()
app = Flask(__name__)


@app.route("/", methods=["POST"])
def predictions():
    try:
        sbuf = io.StringIO()
        _json_response = OrderedDict()
        global size, class_names, yolo, classes
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
                content = request.json

                logging.info("Image resolution %s"%content["Image_resolution"])
                logging.info("Bytes sent %s" %content["bytes_sent"])
                b64_string=content['Image']

                # b64_string = request.data.decode()
                logging.info("Size of Image recieved")
                logging.info((len(b64_string)*3)/4)
                # sbuf.write(base64.b64decode(request.data.decode()))
                # nparr = np.fromstring(request.data, np.uint8)
                bytes_image = io.BytesIO(base64.b64decode(b64_string))
                image1 = imread(bytes_image)
                # np_im = np.array(image1)
                img_in = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                img_in = tf.expand_dims(img_in, 0)
                img_in = transform_images(img_in, size)
                t1 = time.time()
                boxes, scores, classes, nums = yolo(img_in)

                t2 = time.time()
                logging.info('time: {}'.format(t2 - t1))

                logging.info('detections:')
                for i in range(nums[0]):
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                       np.array(scores[0][i]),
                                                       np.array(boxes[0][i])))
                proto_tensor = tf.make_tensor_proto(scores[0])
                scores=tf.make_ndarray(proto_tensor)
                proto_tensor = tf.make_tensor_proto(boxes[0])
                boxes = tf.make_ndarray(proto_tensor)
                proto_tensor = tf.make_tensor_proto(classes[0])
                classes = tf.make_ndarray(proto_tensor)
                if nums[0] is not None :
                    if nums[0] > 0:
                        max_result_roi = boxes[scores.argmax()]
                        max_score = int(scores.argmax())
                        max_class_name = class_names[int(classes[scores.argmax()])]
                        max_result_roi_string = ', '.join(map(str, max_result_roi))
                    if nums[0] > 1:
                        pass
                        #Will be coded when classes get to increase
                _json_response = OrderedDict()
                _json_response['first_score'] = max_score
                _json_response['first_class'] = max_class_name
                _json_response['first_roi'] = max_result_roi_string
                _json_response['other_scores'] = second_max_score

                _json_response['other_classes'] = second_max_class_name

                _json_response['other_rois'] = second_max_roi_string
                _json_response['time_sent'] = time.strftime('%Y-%m-%d %H:%M:%S')
                _json_response['Success']=True
                _json_response['Exception']='None'
                logging.info(_json_response)
            return json.dumps(_json_response, sort_keys=True)
    except Exception as e:
        _json_response = OrderedDict()
        _json_response['first_score'] = max_score
        _json_response['first_class'] = max_class_name
        _json_response['first_roi'] = max_result_roi_string
        _json_response['other_scores'] = second_max_score

        _json_response['other_classes'] = second_max_class_name

        _json_response['other_rois'] = second_max_roi_string
        _json_response['time_sent'] = str(time.strftime('%Y-%m-%d %H:%M:%S'))
        _json_response['Success'] = False
        _json_response['Exception'] = str(e)
        logging.info("Exception occured in flask. Exception: %s"%e)
        return json.dumps(_json_response, sort_keys=True)


if __name__ == '__main__':
    try:
        app.run('0.0.0.0')
    except SystemExit:
        pass


