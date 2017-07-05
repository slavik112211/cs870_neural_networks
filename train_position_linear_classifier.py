import math
import glob
import os
import numpy as np
from PIL import Image
import shutil
import cPickle
import pdb
from datetime import datetime

# python -m pdb train_position_linear_classifier.py
def main():
  print 'Started: ' + str(datetime.now())
  filelist = sorted(glob.glob('./trains_images_dataset/preprocessed/*.jpg'))
  label_images_dict = dict()
  trains_filters = get_trains_filters()

  for filter_name, trains_filter in trains_filters.iteritems():
    label_images_dict[filter_name] = filter_images(filter_name, filelist, trains_filter)
    print filter_name + ': ' + str(label_images_dict[filter_name])

  with open('./label_images_dict', 'wb') as label_images_file:
    cPickle.dump(label_images_dict, label_images_file)

  pdb.set_trace()
  image_labels_dict = generate_image_labels_dict(label_images_dict)
  create_unclassified_images_category(filelist, image_labels_dict)
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
  filtered_images_dir = './trains_images_dataset/preprocessed/unclass/'
  if os.path.exists(filtered_images_dir): shutil.rmtree(filtered_images_dir)
  os.makedirs(filtered_images_dir)

  unclassified_images_list = []
  for filename in filelist:
    image_file = os.path.basename(filename)
    if image_file not in image_labels_dict:
      unclassified_images_list.append(image_file)
      image_labels_dict[image_file] = dict()
      image_labels_dict[image_file][0] = 'unclass'
      shutil.copy2(filename, filtered_images_dir)

  with open(filtered_images_dir+'/images_list', 'wb') as images_list_file:
    cPickle.dump(unclassified_images_list, images_list_file)
  pdb.set_trace()
  print "Unclassified " + str(len(unclassified_images_list)) + ": " + str(unclassified_images_list)
  return image_labels_dict

def filter_images(filter_name, filelist, trains_filter):
  filtered_images_dir = './trains_images_dataset/preprocessed/'+filter_name
  if os.path.exists(filtered_images_dir): shutil.rmtree(filtered_images_dir)
  os.makedirs(filtered_images_dir)
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
      shutil.copy2(filename, filtered_images_dir)
  with open(filtered_images_dir+'/images_list', 'wb') as images_list_file:
    cPickle.dump(filtered_images_list, images_list_file)
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

# luminosity: total luminosity of pixels within '1531' filter
# my_photo-1531.jpg: has a train,      luminosity: (20,4) 31686 or (20,5) 34903
# my_photo-1622.jpg: has no train,     luminosity: (20,4) 43844 or (20,5) 47334
# my_photo-1532.jpg: has a half-train, luminosity: (20,4) 37417 or (20,5) 40616
# my_photo-71.jpg:   has a grey train, luminosity: (20,4) 42266 or (20,5) 45856

def get_trains_filters():
  return {
   'a0': {'x_center': 102, 'y_center':  41, 'rotation_angle':  0, 'luminosity_threshold': 38000},
   'a1': {'x_center': 147, 'y_center':  42, 'rotation_angle':  0, 'luminosity_threshold': 38000},
   'a2': {'x_center': 183, 'y_center':  42, 'rotation_angle':  0, 'luminosity_threshold': 38000},
   'a3': {'x_center': 228, 'y_center':  43, 'rotation_angle':  0, 'luminosity_threshold': 38000},
   'a4': {'x_center': 271, 'y_center':  44, 'rotation_angle':  0, 'luminosity_threshold': 38000},

   'b0': {'x_center': 306, 'y_center':  51, 'rotation_angle':  27, 'luminosity_threshold': 38000},
   'b1': {'x_center': 333, 'y_center':  76, 'rotation_angle':  58, 'luminosity_threshold': 38000},
   'b2': {'x_center': 342, 'y_center': 100, 'rotation_angle':  77, 'luminosity_threshold': 38000},
   'b3': {'x_center': 345, 'y_center': 125, 'rotation_angle':  90, 'luminosity_threshold': 38000},
   'b4': {'x_center': 342, 'y_center': 152, 'rotation_angle':  99, 'luminosity_threshold': 38000},
   'b5': {'x_center': 301, 'y_center': 204, 'rotation_angle': 160, 'luminosity_threshold': 41000},

   'c0': {'x_center': 277, 'y_center': 210, 'rotation_angle': -3, 'luminosity_threshold': 39000},
   'c1': {'x_center': 246, 'y_center': 210, 'rotation_angle':  0, 'luminosity_threshold': 38000},
   'c2': {'x_center': 210, 'y_center': 210, 'rotation_angle':  0, 'luminosity_threshold': 38000},
   'c3': {'x_center': 182, 'y_center': 208, 'rotation_angle':  0, 'luminosity_threshold': 38000},
   'c4': {'x_center': 138, 'y_center': 207, 'rotation_angle':  0, 'luminosity_threshold': 38000},
   'c5': {'x_center':  98, 'y_center': 206, 'rotation_angle':  0, 'luminosity_threshold': 39000},

   'd0': {'x_center':  78, 'y_center': 205, 'rotation_angle':  13, 'luminosity_threshold': 40000},
   'd1': {'x_center':  52, 'y_center': 195, 'rotation_angle':  34, 'luminosity_threshold': 40000},
   'd2': {'x_center':  40, 'y_center': 182, 'rotation_angle':  52, 'luminosity_threshold': 38000},
   'd3': {'x_center':  26, 'y_center': 160, 'rotation_angle':  72, 'luminosity_threshold': 38000},
   'd4': {'x_center':  25, 'y_center': 121, 'rotation_angle':  90, 'luminosity_threshold': 38000},
   'd5': {'x_center':  30, 'y_center':  84, 'rotation_angle': -70, 'luminosity_threshold': 38000},
   'd6': {'x_center':  49, 'y_center':  57, 'rotation_angle': -45, 'luminosity_threshold': 38000},
   'd7': {'x_center':  78, 'y_center':  41, 'rotation_angle': -11, 'luminosity_threshold': 38000},

   'e0': {'x_center': 127, 'y_center':  48, 'rotation_angle':  21, 'luminosity_threshold': 38000},
   'e1': {'x_center': 155, 'y_center':  67, 'rotation_angle':  46, 'luminosity_threshold': 38000},
   'e2': {'x_center': 168, 'y_center':  85, 'rotation_angle':  62, 'luminosity_threshold': 38000},
   'e3': {'x_center': 176, 'y_center': 119, 'rotation_angle':  90, 'luminosity_threshold': 38000},
   'e4': {'x_center': 172, 'y_center': 144, 'rotation_angle': -75, 'luminosity_threshold': 40000},
   'e5': {'x_center': 156, 'y_center': 175, 'rotation_angle': -50, 'luminosity_threshold': 43000},
   'e6': {'x_center': 129, 'y_center': 197, 'rotation_angle': -27, 'luminosity_threshold': 43000},
   'e7': {'x_center': 113, 'y_center': 203, 'rotation_angle': -13, 'luminosity_threshold': 41000},

   'f0': {'x_center': 229, 'y_center': 204, 'rotation_angle':  25, 'luminosity_threshold': 40000},
   'f1': {'x_center': 195, 'y_center': 180, 'rotation_angle':  54, 'luminosity_threshold': 43000},
   'f2': {'x_center': 181, 'y_center': 155, 'rotation_angle':  70, 'luminosity_threshold': 42000},
   'f3': {'x_center': 188, 'y_center':  86, 'rotation_angle': -58, 'luminosity_threshold': 39000},
   'f4': {'x_center': 206, 'y_center':  65, 'rotation_angle': -35, 'luminosity_threshold': 40000},
   'f5': {'x_center': 236, 'y_center':  49, 'rotation_angle': -20, 'luminosity_threshold': 38000},

   'g0': {'x_center':  34, 'y_center':  62, 'rotation_angle': -60, 'luminosity_threshold': 38000},
   'g1': {'x_center':  47, 'y_center':  44, 'rotation_angle': -44, 'luminosity_threshold': 38000},
   'g2': {'x_center':  78, 'y_center':  26, 'rotation_angle': -14, 'luminosity_threshold': 38000},
   'g3': {'x_center':  98, 'y_center':  25, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'g4': {'x_center': 124, 'y_center':  25, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'g5': {'x_center': 150, 'y_center':  26, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'g6': {'x_center': 214, 'y_center':  29, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'g7': {'x_center': 246, 'y_center':  28, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'g8': {'x_center': 281, 'y_center':  32, 'rotation_angle':  17, 'luminosity_threshold': 38000},
   'g9': {'x_center': 312, 'y_center':  47, 'rotation_angle':  38, 'luminosity_threshold': 38000},
  'g10': {'x_center': 328, 'y_center':  65, 'rotation_angle':  58, 'luminosity_threshold': 38000},

   'h0': {'x_center': 322, 'y_center': 197, 'rotation_angle': -53, 'luminosity_threshold': 45000},
   'h1': {'x_center': 300, 'y_center': 214, 'rotation_angle': -31, 'luminosity_threshold': 43000},
   'h2': {'x_center': 278, 'y_center': 224, 'rotation_angle': -14, 'luminosity_threshold': 42000},
   'h3': {'x_center': 261, 'y_center': 226, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'h4': {'x_center': 232, 'y_center': 225, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'h5': {'x_center': 204, 'y_center': 225, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'h6': {'x_center': 177, 'y_center': 223, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'h7': {'x_center': 145, 'y_center': 223, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'h8': {'x_center': 114, 'y_center': 223, 'rotation_angle':   0, 'luminosity_threshold': 38000},
   'h9': {'x_center':  85, 'y_center': 222, 'rotation_angle':   7, 'luminosity_threshold': 42000},
  'h10': {'x_center':  45, 'y_center': 205, 'rotation_angle':  42, 'luminosity_threshold': 38000},
  'h11': {'x_center':  30, 'y_center': 183, 'rotation_angle':  63, 'luminosity_threshold': 38000},

   'i0': {'x_center': 114, 'y_center': 23, 'rotation_angle':  -10, 'luminosity_threshold': 38000},
   'i1': {'x_center': 139, 'y_center': 16, 'rotation_angle':  -19, 'luminosity_threshold': 38000},
   'i2': {'x_center': 164, 'y_center': 12, 'rotation_angle':    0, 'luminosity_threshold': 38000},
   'i3': {'x_center': 181, 'y_center': 13, 'rotation_angle':    0, 'luminosity_threshold': 38000},
   'i4': {'x_center': 198, 'y_center': 13, 'rotation_angle':    6, 'luminosity_threshold': 38000},
   'i5': {'x_center': 218, 'y_center': 18, 'rotation_angle':   22, 'luminosity_threshold': 38000},
   'i6': {'x_center': 237, 'y_center': 25, 'rotation_angle':   20, 'luminosity_threshold': 38000},

   'j0': {'x_center': 225, 'y_center':  13, 'rotation_angle':   0, 'luminosity_threshold': 40000},
   'j1': {'x_center': 250, 'y_center':  13, 'rotation_angle':   0, 'luminosity_threshold': 40000},
   'j2': {'x_center': 278, 'y_center':  16, 'rotation_angle':  12, 'luminosity_threshold': 40000},
   'j3': {'x_center': 301, 'y_center':  24, 'rotation_angle':  26, 'luminosity_threshold': 40000},
   'j4': {'x_center': 332, 'y_center':  38, 'rotation_angle':  25, 'luminosity_threshold': 40000},
   'j5': {'x_center': 364, 'y_center':  54, 'rotation_angle':  25, 'luminosity_threshold': 40000},
   'j6': {'x_center': 390, 'y_center':  76, 'rotation_angle':  49, 'luminosity_threshold': 40000},
   'j7': {'x_center': 404, 'y_center': 107, 'rotation_angle':  81, 'luminosity_threshold': 43000},
   'j8': {'x_center': 405, 'y_center': 125, 'rotation_angle':  90, 'luminosity_threshold': 43000},
   'j9': {'x_center': 403, 'y_center': 160, 'rotation_angle': -73, 'luminosity_threshold': 40000},
  'j10': {'x_center': 395, 'y_center': 176, 'rotation_angle': -59, 'luminosity_threshold': 43000},
  'j11': {'x_center': 370, 'y_center': 201, 'rotation_angle': -30, 'luminosity_threshold': 43000},
  'j12': {'x_center': 337, 'y_center': 216, 'rotation_angle': -23, 'luminosity_threshold': 45000},
  'j13': {'x_center': 313, 'y_center': 224, 'rotation_angle':  -9, 'luminosity_threshold': 45000},
  'j14': {'x_center': 287, 'y_center': 226, 'rotation_angle':   0, 'luminosity_threshold': 43000}
  }

main()