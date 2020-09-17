# Glasses detection and recognition implemented in TensorFlow 2.0



## Key Features

Mainly uses implementation of following repo
In case of any issue refer to the original implementation

The code is updated and made functional according to Tensorflow 2.0
https://github.com/zzh8829/yolov3-tf2


## Usage

### Installation

```

#### Pip

```bash
pip install -r requirements.txt
```

### Nvidia and cuda installations (For GPU)

Please install cuda 10.0 and Cudnn 7.6 for this repo


### Convert pre-trained Darknet weights

First download the pre-trained weights of Darknet and convert them to tf model

```bash
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny
```


### Conversion of VOC dataset annotated to tf.data.record
Dataset can found in the following link:
```
https://drive.google.com/drive/folders/14eYZa4FpspzGlYm6NP5hihC_reA6m3s_?usp=sharing
```

Place dataset in the format of VOC2009 in the data directory and convert using following commands

``` bash
python tools/voc2012.py --data_dir './data/GlassesData'  --split train  --output_file ./data/voc_train.tfrecord

python tools/voc2012.py \ --data_dir './data/GlassesData' \ --split val \ --output_file ./data/voc_val.tfrecord
```
### Training

I have created a complete tutorial on how to train from scratch using the VOC2012 Dataset.
See the documentation here https://github.com/zzh8829/yolov3-tf2/blob/master/docs/training_voc.md

For customzied training, you need to generate tfrecord following the TensorFlow Object Detection API.
For example you can use [Microsoft VOTT](https://github.com/Microsoft/VoTT) to generate such dataset.
You can also use this [script](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py) to create the pascal voc dataset.

Example commend line arguments for training
``` bash
python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --mode eager_tf --transfer fine_tune

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --mode fit --transfer none

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --mode fit --transfer no_output

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 10 --mode eager_fit --transfer fine_tune --weights ./checkpoints/yolov3-tiny.tf --tiny
```
My command
```bash
python train.py --dataset ./data/voc_train.tfrecord --val_dataset ./data/voc_val.tfrecord --classes ./data/voc2012.names --num_classes 2 --mode fit --transfer darknet --batch_size 16 --epochs 20 --weights ./checkpoints/yolov3.tf --weights_num_classes 80

```

## Testing
For testing the model can be found on the following link: 

-For now the checkpoint files are larger and can't be uploaded in git. Please download it from following link 
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I9ugHr_dnQD00zMeOKgW26BNqXi6OEwn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1I9ugHr_dnQD00zMeOKgW26BNqXi6OEwn" -O yolov3_train_11.tf.zip && rm -rf /tmp/cookies.txt
```
4 class detector model files
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dMDY56iVm9KrnIC1u7H3Ti958hVsX25G' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dMDY56iVm9KrnIC1u7H3Ti958hVsX25G" -O yolov3_train_22.tf.zip && rm -rf /tmp/cookies.txt
```
put it in checkpoints folder

```
-->yolov3_tf2
   |->checkpoints
     |->checkpoint (file)
```

Also rename the model name in checkpoint file accoding to the epoch number and files
For example

For files 

#### yolov3_train_11.tf.data-00000-of-00002, yolov3_train_11.tf.index, yolov3_train_11.tf.data-00001-of-00002

Specify 
#### yolov3_train_11.tf

```bash
# yolov3
python detect.py --image ./data/meme.jpg

# yolov3-tiny
python detect.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image ./data/street.jpg

# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video path_to_file.mp4 --weights ./checkpoints/yolov3-tiny.tf --tiny

# video file with output
python detect_video.py --video path_to_file.mp4 --output ./output.avi
```
## test results 
These are the test results of two classes Oakley M2 and Oakley2

| F1 Score  | 0.62 |
|-----------|------|
| Precision | 0.64 |
| Recall    | 0.61 |

These metrics are measured using this repository. Will upload the code and method of it later.
https://github.com/penny4860/tf2-eager-yolo3

## Deployment Notes
For deployment refer to the following repo: 
https://github.com/arehmanAzam/detector20_deploy

## Command Line Args Reference

```bash
convert.py:
  --output: path to output
    (default: './checkpoints/yolov3.tf')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './data/yolov3.weights')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)

detect.py:
  --classes: path to classes file
    (default: './data/coco.names')
  --image: path to input image
    (default: './data/girl.png')
  --output: path to output image
    (default: './output.jpg')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov3.tf')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)

detect_video.py:
  --classes: path to classes file
    (default: './data/coco.names')
  --video: path to input video (use 0 for cam)
    (default: './data/video.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov3.tf')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)

train.py:
  --batch_size: batch size
    (default: '8')
    (an integer)
  --classes: path to classes file
    (default: './data/coco.names')
  --dataset: path to dataset
    (default: '')
  --epochs: number of epochs
    (default: '2')
    (an integer)
  --learning_rate: learning rate
    (default: '0.001')
    (a number)
  --mode: <fit|eager_fit|eager_tf>: fit: model.fit, eager_fit: model.fit(run_eagerly=True), eager_tf: custom GradientTape
    (default: 'fit')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)
  --size: image size
    (default: '416')
    (an integer)
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --transfer: <none|darknet|no_output|frozen|fine_tune>: none: Training from scratch, darknet: Transfer darknet, no_output: Transfer all but output, frozen: Transfer and freeze all,
    fine_tune: Transfer all and freeze darknet only
    (default: 'none')
  --val_dataset: path to validation dataset
    (default: '')
  --weights: path to weights file
    (default: './checkpoints/yolov3.tf')

