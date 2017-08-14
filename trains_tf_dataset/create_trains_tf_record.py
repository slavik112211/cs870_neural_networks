import hashlib
import io
import logging
import os
import random
import re

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

import _pickle as cPickle
import pdb
from scipy import misc
import numpy as np

from training_categories import get_trains_filters

"""
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_pets.md
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md

# From the tensorflow/models/ directory
export TENSORFLOW_DATASETS=/home/u3/vderevyanko/workspace/cs870_neural_networks/trains_tf_dataset
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${TENSORFLOW_DATASETS}/data/ssd_inception_v2_trains.config \
    --train_dir=${TENSORFLOW_DATASETS}/train/

# From the tensorflow/models/ directory
python3 object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${TENSORFLOW_DATASETS}/data/ssd_inception_v2_trains.config \
    --checkpoint_dir=${TENSORFLOW_DATASETS}/train/ \
    --eval_dir=${TENSORFLOW_DATASETS}/eval/

tensorboard --logdir=${TENSORFLOW_DATASETS}

"""

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 240
categories_list = sorted(get_trains_filters().keys())

def save_trains_tf_dataset():
  image_labels_dict = {}
  with open('./image_labels_dict', 'rb') as image_labels_file:
    image_labels_dict = cPickle.load(image_labels_file)
  categories_map_dict = label_map_util.get_label_map_dict('./trains_labelmap') # has just 1 category: "train"

  pdb.set_trace()
  data_dir = './trains_images_noise_rgb/'

  img_filenames = [ filename for filename in os.listdir(data_dir) if filename.endswith('.jpg') ]
  total_labelled_images = len(img_filenames) # 14360 with noise / 1436 without noise
  test_set_size = int(total_labelled_images / 6) # 2393 / 239
  train_set_size = total_labelled_images - test_set_size  # 11967 / 1197
  test_set_filenames = random.sample(img_filenames, test_set_size)

  training_data = []; test_data = []
  random.shuffle(img_filenames)

  writer_training_data = tf.python_io.TFRecordWriter('./training_data.record')
  writer_valid_data = tf.python_io.TFRecordWriter('./valid_data.record')

  for index, image_file in enumerate(img_filenames):
    if index % 10 == 0: print(str(index) + " / " + str(len(img_filenames)))

    general_filename = image_file.split('.')[0][:-2] + "." + image_file.split('.')[1]
    categories_map = image_labels_dict[general_filename]
    category = categories_map[min(categories_map)]
    train_coordinates = get_trains_filters()[category]

    tf_example = dict_to_tf_example(train_coordinates, categories_map_dict, image_file, data_dir)

    if image_file not in test_set_filenames:
      writer_training_data.write(tf_example.SerializeToString())
    else:
      writer_valid_data.write(tf_example.SerializeToString())

  writer_training_data.close()
  writer_valid_data.close()

def dict_to_tf_example(train_coordinates, categories_map_dict, image_filename, data_dir):
  img_path = os.path.join(data_dir, image_filename)
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = IMAGE_WIDTH
  height = IMAGE_HEIGHT
  xmin1 = train_coordinates['x_center'] - 25
  xmax1 = train_coordinates['x_center'] + 25
  ymin1 = train_coordinates['y_center'] - 25
  ymax1 = train_coordinates['y_center'] + 25

  if(xmin1 < 0): xmin1 = 0
  if(xmax1 > IMAGE_WIDTH): xmax1 = IMAGE_WIDTH
  if(ymin1 < 0): ymin1 = 0
  if(ymax1 > IMAGE_HEIGHT): ymax1 = IMAGE_HEIGHT

  xmin = []; ymin = []; xmax = []; ymax = []; classes = []; classes_text = []

  xmin.append(float(xmin1) / width)
  xmax.append(float(xmax1) / width)
  ymin.append(float(ymin1) / height)
  ymax.append(float(ymax1) / height)

  class_name = 'train'
  classes_text.append(class_name.encode('utf8'))
  classes.append(categories_map_dict[class_name])

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example

def convert_grayscale_to_rgb():
  images_dir = './trains_images_noise/'
  images_rgb_dir = './trains_images_noise_rgb/'
  img_filenames = [ filename for filename in os.listdir(images_dir) if filename.endswith('.jpg') ]

  rgb = np.zeros((255, 255, 3), dtype=np.uint8)

  for index, image_file in enumerate(img_filenames):
    if index % 10 == 0: print(str(index) + " / " + str(len(img_filenames)))
    image_matrix = misc.imread(images_dir+image_file)
    image_matrix = np.dstack([image_matrix.astype(np.uint8)] * 3)
    misc.imsave(images_rgb_dir+image_file, image_matrix)

save_trains_tf_dataset()