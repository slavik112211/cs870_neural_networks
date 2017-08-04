import math
import glob
import os
import numpy as np
from PIL import Image
import shutil
import cPickle
import pdb
from datetime import datetime

from training_categories import get_trains_filters

# python -m pdb train_position_linear_classifier.py
def main():
  print 'Started: ' + str(datetime.now())
  filelist = sorted(glob.glob('./trains_images_preprocessed/*.jpg'))
  label_images_dict = dict()
  trains_filters = get_trains_filters()

  for filter_name, trains_filter in trains_filters.iteritems():
    label_images_dict[filter_name] = filter_images(filter_name, filelist, trains_filter)
    print filter_name + ': ' + str(len(label_images_dict[filter_name]))

  with open('./label_images_dict', 'wb') as label_images_file:
    cPickle.dump(label_images_dict, label_images_file)

  # pdb.set_trace()
  image_labels_dict = generate_image_labels_dict(label_images_dict)
  create_unclassified_images_category(filelist, image_labels_dict)
  copy_categorized_images(image_labels_dict)
  with open('./image_labels_dict', 'wb') as image_labels_file:
    cPickle.dump(image_labels_dict, image_labels_file)
  print 'Finished: ' + str(datetime.now())

def generate_image_labels_dict(label_images_dict):
  image_labels_dict = dict()
  for label, images in label_images_dict.iteritems():
    for image_file, filter_luminosity in images.iteritems():
      if image_file in image_labels_dict:
        image_labels_dict[image_file][filter_luminosity] = label
        print "Duplicate label for image " + image_file + ": " + str(image_labels_dict[image_file])
      else:
        image_labels_dict[image_file] = dict()
        image_labels_dict[image_file][filter_luminosity] = label
  return image_labels_dict

def create_unclassified_images_category(filelist, image_labels_dict):
  unclassified_images_list = []
  for filename in filelist:
    image_file = os.path.basename(filename)
    if image_file not in image_labels_dict:
      unclassified_images_list.append(image_file)
      # image_labels_dict[image_file] = dict()
      # image_labels_dict[image_file][0] = 'unclass'

  with open('./unclassified', 'wb') as unclassified_file:
    cPickle.dump(unclassified_images_list, unclassified_file)
  print "Unclassified files: " + str(len(unclassified_images_list))
  return image_labels_dict

def copy_categorized_images(image_labels_dict):
  images_dir = './trains_images_preprocessed/'
  for image_file, categories_map in image_labels_dict.iteritems():
    # take the category that has lowest luminosity value, {30000=>'a1', 31000=>'a2'} should return 'a1'
    category = categories_map[min(categories_map)]
    category_images_dir = images_dir + str(category) + '/'
    if not os.path.exists(category_images_dir): os.makedirs(category_images_dir)
    shutil.copy2(images_dir + image_file, category_images_dir)


def filter_images(filter_name, filelist, trains_filter):
  filter_pixels = get_filter_pixels(trains_filter)

  filtered_images_list = dict()
  for filename in filelist:
    image = Image.open(filename)
    # http://pillow.readthedocs.io/en/3.4.x/reference/Image.html
    image_gray = image.convert('L') # grayscale, pixel values: 0 - black, 255 - white

    total_luminosity = 0
    for filter_pixel in filter_pixels:
      total_luminosity += image_gray.getpixel((filter_pixel[0],filter_pixel[1]))
    # if(os.path.basename(filename) == 'my_photo-1864.jpg'): pdb.set_trace()
    if(total_luminosity < trains_filter['luminosity_threshold']): # Main condition for filtering an image
      filtered_images_list[os.path.basename(filename)] = total_luminosity
  return filtered_images_list

def get_filter_pixels(trains_filter):
  # train filter: rectangle 40x10 pixels
  filter_x_start = 0; filter_x_end = 39; filter_x_center = 20;
  filter_y_start = 0; filter_y_end = 9;  filter_y_center = 5;

  filter_pixels = []

  transform_matrix = get_transform_matrix(filter_x_center, filter_y_center,
      trains_filter['x_center'], trains_filter['y_center'], trains_filter['rotation_angle'])

  for i in range(filter_y_start, filter_y_end+1):
    for j in range(filter_x_start, filter_x_end+1):
      point = np.array([j, i, 1])
      point_after_transform = np.dot(transform_matrix, point)
      filter_pixels.append((int(round(point_after_transform[0])), int(round(point_after_transform[1]))))
  
  return filter_pixels

def get_transform_matrix(filter_x_center, filter_y_center,
    filter_x_center_after_transform, filter_y_center_after_transform, rotation_angle):

  angle_rad = (rotation_angle * math.pi) / 180
  cos_a = math.cos(angle_rad)
  sin_a = math.sin(angle_rad)

  move_to_origin_matrix = np.array(
    [[1, 0, -filter_x_center],
     [0, 1, -filter_y_center],
     [0, 0,              1]])
  rotate_matrix = np.array(
    [[cos_a, -sin_a, 0],
     [sin_a,  cos_a, 0],
     [0,          0, 1]])
  move_from_origin_matrix = np.array(
    [[1, 0, filter_x_center_after_transform],
     [0, 1, filter_y_center_after_transform],
     [0, 0,                             1]])

  # https://en.wikipedia.org/wiki/Transformation_matrix#Composing_and_inverting_transformations
  # B*(A*x) = (B*A) * x (where x - vector, A and B - matrices)
  # Multiplication is done in the opposite order from the English sentence: the matrix of "A followed by B" is BA, not AB.
  return move_from_origin_matrix.dot(rotate_matrix).dot(move_to_origin_matrix)

# width,height = image_gray.size
# total=0
# for x in range(0,width):
#   for y in range(0,height):
#     total += image_gray.getpixel((x,y))
# mean = total / (width * height)

main()