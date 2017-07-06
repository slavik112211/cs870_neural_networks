import cPickle
import numpy as np
from scipy import misc
import random
import pdb

import network
from training_categories import get_trains_filters

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 240
categories_list = sorted(get_trains_filters().keys())
INPUT_LAYER_SIZE = IMAGE_WIDTH*IMAGE_HEIGHT # 115200
OUTPUT_LAYER_SIZE = len(categories_list)    # 88

def main():
  pdb.set_trace()
  training_data, test_data = load_dataset()
  net = network.Network([INPUT_LAYER_SIZE, 200, OUTPUT_LAYER_SIZE])
  net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

def load_dataset():
  image_labels_dict = {}
  with open('./image_labels_dict', 'rb') as image_labels_file:
    image_labels_dict = cPickle.load(image_labels_file)

  images_dir = './trains_images_dataset/preprocessed/'
  total_labelled_images = len(image_labels_dict) # 1436
  test_set_size = int(total_labelled_images / 6) # 239
  train_set_size = total_labelled_images - test_set_size  # 1197
  test_set_filenames = random.sample(image_labels_dict, test_set_size)

  training_data = []; test_data = []
  for image_file, categories_map in image_labels_dict.iteritems():
    image_matrix = misc.imread(images_dir+image_file)
    image_matrix = np.reshape(image_matrix, (IMAGE_WIDTH*IMAGE_HEIGHT, 1))

    # pick category with lowest luminosity value: {30000=>'a1', 31000=>'a2'}
    category = categories_map[min(categories_map)]
    category_id = categories_list.index(category)
    if image_file not in test_set_filenames: 
      training_data.append((image_matrix, vectorized_output(category_id)))
    else:
      test_data.append((image_matrix, category_id))

  return (training_data, test_data)

def vectorized_output(category_id):
    e = np.zeros((len(categories_list), 1))
    e[category_id] = 1.0
    return e

main()
