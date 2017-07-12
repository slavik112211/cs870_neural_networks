import cPickle
import numpy as np
from scipy import misc
import random
import pdb

from training_categories import get_trains_filters

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 240
categories_list = sorted(get_trains_filters().keys())
INPUT_LAYER_SIZE = IMAGE_WIDTH*IMAGE_HEIGHT # 115200
# OUTPUT_LAYER_SIZE = len(categories_list)    # 88
OUTPUT_LAYER_SIZE = 10 # a,b,c,d,e,f,g,h,i,j

def load_trains_dataset():
  image_labels_dict = {}
  with open('./image_labels_dict', 'rb') as image_labels_file:
    image_labels_dict = cPickle.load(image_labels_file)

  images_dir = './trains_images_preprocessed/'
  total_labelled_images = len(image_labels_dict) # 1436
  test_set_size = int(total_labelled_images / 6) # 239
  train_set_size = total_labelled_images - test_set_size  # 1197
  test_set_filenames = random.sample(image_labels_dict, test_set_size)

  training_data = []; test_data = []
  for image_file, categories_map in image_labels_dict.iteritems():

    image_matrix = misc.imread(images_dir+image_file)
    image_matrix = np.reshape(image_matrix, (IMAGE_WIDTH*IMAGE_HEIGHT, 1))
    image_matrix = image_matrix.astype(np.float32)
    image_matrix = np.multiply(image_matrix, 1.0 / 255.0) # Convert from [0, 255] -> [0.0, 1.0]

    # pick category with lowest luminosity value: {30000=>'a1', 31000=>'a2'}
    category = categories_map[min(categories_map)]
    # category_id = categories_list.index(category)
    category_id = get_united_category_id(category)
    if image_file not in test_set_filenames: 
      training_data.append((image_matrix, vectorized_output(category_id)))
    else:
      test_data.append((image_matrix, category_id))

  pdb.set_trace()
  return (training_data, test_data)

def vectorized_output(category_id):
    e = np.zeros((OUTPUT_LAYER_SIZE, 1))
    e[category_id] = 1.0
    return e

def get_united_category_id(category_name):
  category_id = -1
  if   'a' in category_name: category_id = 0
  elif 'b' in category_name: category_id = 1
  elif 'c' in category_name: category_id = 2
  elif 'd' in category_name: category_id = 3
  elif 'e' in category_name: category_id = 4
  elif 'f' in category_name: category_id = 5
  elif 'g' in category_name: category_id = 6
  elif 'h' in category_name: category_id = 7
  elif 'i' in category_name: category_id = 8
  elif 'j' in category_name: category_id = 9

  return category_id

def get_category_id(category_name):
  category_id = -1
  if   'a01' in category_name: category_id = 0
  elif 'a02' in category_name: category_id = 1
  elif 'a03' in category_name: category_id = 2
  elif 'a04' in category_name: category_id = 3
  elif 'a05' in category_name: category_id = 4
  elif 'b01' in category_name: category_id = 5
  elif 'b02' in category_name: category_id = 6
  elif 'b03' in category_name: category_id = 7
  elif 'b04' in category_name: category_id = 8
  elif 'b05' in category_name: category_id = 9
  elif 'b06' in category_name: category_id = 10
  elif 'b07' in category_name: category_id = 11
  elif 'c01' in category_name: category_id = 12
  elif 'c02' in category_name: category_id = 13
  elif 'c03' in category_name: category_id = 14
  elif 'c04' in category_name: category_id = 15
  elif 'c05' in category_name: category_id = 16
  elif 'c06' in category_name: category_id = 17
  elif 'd01' in category_name: category_id = 18
  elif 'd02' in category_name: category_id = 19
  elif 'd03' in category_name: category_id = 20
  elif 'd04' in category_name: category_id = 21
  elif 'd05' in category_name: category_id = 22
  elif 'd06' in category_name: category_id = 23
  elif 'd07' in category_name: category_id = 24
  elif 'd08' in category_name: category_id = 25
  elif 'd09' in category_name: category_id = 26
  elif 'e01' in category_name: category_id = 27
  elif 'e02' in category_name: category_id = 28
  elif 'e03' in category_name: category_id = 29
  elif 'e04' in category_name: category_id = 30
  elif 'e05' in category_name: category_id = 31
  elif 'e06' in category_name: category_id = 32
  elif 'e07' in category_name: category_id = 33
  elif 'e08' in category_name: category_id = 34
  elif 'e09' in category_name: category_id = 35
  elif 'f01' in category_name: category_id = 36
  elif 'f02' in category_name: category_id = 37
  elif 'f03' in category_name: category_id = 38
  elif 'f04' in category_name: category_id = 39
  elif 'f05' in category_name: category_id = 40
  elif 'f06' in category_name: category_id = 41
  elif 'g01' in category_name: category_id = 42
  elif 'g02' in category_name: category_id = 43
  elif 'g03' in category_name: category_id = 44
  elif 'g04' in category_name: category_id = 45
  elif 'g05' in category_name: category_id = 46
  elif 'g06' in category_name: category_id = 47
  elif 'g07' in category_name: category_id = 48
  elif 'g08' in category_name: category_id = 49
  elif 'g09' in category_name: category_id = 50
  elif 'g10' in category_name: category_id = 51
  elif 'g11' in category_name: category_id = 52
  elif 'h01' in category_name: category_id = 53
  elif 'h02' in category_name: category_id = 54
  elif 'h03' in category_name: category_id = 55
  elif 'h04' in category_name: category_id = 56
  elif 'h05' in category_name: category_id = 57
  elif 'h06' in category_name: category_id = 58
  elif 'h07' in category_name: category_id = 59
  elif 'h08' in category_name: category_id = 60
  elif 'h09' in category_name: category_id = 61
  elif 'h10' in category_name: category_id = 62
  elif 'h11' in category_name: category_id = 63
  elif 'h12' in category_name: category_id = 64
  elif 'h13' in category_name: category_id = 65
  elif 'i01' in category_name: category_id = 66
  elif 'i02' in category_name: category_id = 67
  elif 'i03' in category_name: category_id = 68
  elif 'i04' in category_name: category_id = 69
  elif 'i05' in category_name: category_id = 70
  elif 'i06' in category_name: category_id = 71
  elif 'i07' in category_name: category_id = 72
  elif 'j01' in category_name: category_id = 73
  elif 'j02' in category_name: category_id = 74
  elif 'j03' in category_name: category_id = 75
  elif 'j04' in category_name: category_id = 76
  elif 'j05' in category_name: category_id = 77
  elif 'j06' in category_name: category_id = 78
  elif 'j07' in category_name: category_id = 79
  elif 'j08' in category_name: category_id = 80
  elif 'j09' in category_name: category_id = 81
  elif 'j10' in category_name: category_id = 82
  elif 'j11' in category_name: category_id = 83
  elif 'j12' in category_name: category_id = 84
  elif 'j13' in category_name: category_id = 85
  elif 'j14' in category_name: category_id = 86
  elif 'j15' in category_name: category_id = 87

  return category_id