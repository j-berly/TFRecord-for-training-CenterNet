# TFRecord for training CenterNet
Using keypoints for classfier and regression, like CenterNet. 

Remove ct* in tf_example_decoder.py for CornerNet.
## Info
This code uses to create and decode tfrecord which saved keypoints instead of bboxes.

Only support COCO format dataset now.

Given thanks to https://github.com/Duankaiwen/CenterNet and https://github.com/google/automl
## Environment
tensorflow 2.1.0 (change import part if you use tf v1)

numpy 1.17.0

opencv-python  4.2.0
## Create TFRecord
set parameters in config.json

    python create_coco_kp_tfrecord.py

## Decode TFRecord

    python tf_example_decoder.py

