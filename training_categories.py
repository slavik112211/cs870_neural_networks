# luminosity: total luminosity of pixels within '1531' filter
# my_photo-1531.jpg: has a train,      luminosity: (20,4) 31686 or (20,5) 34903
# my_photo-1622.jpg: has no train,     luminosity: (20,4) 43844 or (20,5) 47334
# my_photo-1532.jpg: has a half-train, luminosity: (20,4) 37417 or (20,5) 40616
# my_photo-71.jpg:   has a grey train, luminosity: (20,4) 42266 or (20,5) 45856

def get_trains_filters():
  return {
   'a01': {'x_center': 102, 'y_center':  41, 'rotation_angle':  0, 'luminosity_threshold': 39000},
   'a02': {'x_center': 147, 'y_center':  42, 'rotation_angle':  0, 'luminosity_threshold': 39000},
   'a03': {'x_center': 183, 'y_center':  42, 'rotation_angle':  0, 'luminosity_threshold': 42000},
   'a04': {'x_center': 228, 'y_center':  43, 'rotation_angle':  0, 'luminosity_threshold': 42000},
   'a05': {'x_center': 271, 'y_center':  44, 'rotation_angle':  0, 'luminosity_threshold': 42000},

   'b01': {'x_center': 306, 'y_center':  51, 'rotation_angle':  27, 'luminosity_threshold': 38000},
   'b02': {'x_center': 333, 'y_center':  76, 'rotation_angle':  58, 'luminosity_threshold': 42000},
   'b03': {'x_center': 342, 'y_center': 100, 'rotation_angle':  77, 'luminosity_threshold': 42000},
   'b04': {'x_center': 345, 'y_center': 125, 'rotation_angle':  90, 'luminosity_threshold': 42000},
   'b05': {'x_center': 342, 'y_center': 152, 'rotation_angle':  99, 'luminosity_threshold': 42000},
   'b06': {'x_center': 318, 'y_center': 195, 'rotation_angle': -42, 'luminosity_threshold': 42000},
   'b07': {'x_center': 301, 'y_center': 204, 'rotation_angle': 160, 'luminosity_threshold': 42000},

   'c01': {'x_center': 277, 'y_center': 210, 'rotation_angle': -3, 'luminosity_threshold': 42000},
   'c02': {'x_center': 246, 'y_center': 210, 'rotation_angle':  0, 'luminosity_threshold': 42000},
   'c03': {'x_center': 210, 'y_center': 210, 'rotation_angle':  0, 'luminosity_threshold': 42000},
   'c04': {'x_center': 182, 'y_center': 208, 'rotation_angle':  0, 'luminosity_threshold': 42000},
   'c05': {'x_center': 138, 'y_center': 207, 'rotation_angle':  0, 'luminosity_threshold': 42000},
   'c06': {'x_center':  98, 'y_center': 206, 'rotation_angle':  0, 'luminosity_threshold': 42000},

   'd01': {'x_center':  69, 'y_center': 200, 'rotation_angle':  19, 'luminosity_threshold': 42000},
   'd02': {'x_center':  78, 'y_center': 205, 'rotation_angle':  13, 'luminosity_threshold': 42000},
   'd03': {'x_center':  52, 'y_center': 195, 'rotation_angle':  34, 'luminosity_threshold': 42000},
   'd04': {'x_center':  40, 'y_center': 182, 'rotation_angle':  52, 'luminosity_threshold': 42000},
   'd05': {'x_center':  26, 'y_center': 160, 'rotation_angle':  72, 'luminosity_threshold': 42000},
   'd06': {'x_center':  25, 'y_center': 121, 'rotation_angle':  90, 'luminosity_threshold': 38000},
   'd07': {'x_center':  30, 'y_center':  84, 'rotation_angle': -70, 'luminosity_threshold': 38000},
   'd08': {'x_center':  49, 'y_center':  57, 'rotation_angle': -45, 'luminosity_threshold': 38000},
   'd09': {'x_center':  78, 'y_center':  41, 'rotation_angle': -11, 'luminosity_threshold': 42000},

   'e01': {'x_center': 127, 'y_center':  48, 'rotation_angle':  21, 'luminosity_threshold': 40000},
   'e02': {'x_center': 155, 'y_center':  67, 'rotation_angle':  46, 'luminosity_threshold': 41000},
   'e03': {'x_center': 168, 'y_center':  85, 'rotation_angle':  62, 'luminosity_threshold': 40000},
   'e04': {'x_center': 176, 'y_center': 119, 'rotation_angle':  90, 'luminosity_threshold': 41000},
   'e05': {'x_center': 172, 'y_center': 144, 'rotation_angle': -75, 'luminosity_threshold': 42000},
   'e06': {'x_center': 156, 'y_center': 175, 'rotation_angle': -50, 'luminosity_threshold': 43000},
   'e07': {'x_center': 129, 'y_center': 197, 'rotation_angle': -27, 'luminosity_threshold': 43000},
   'e08': {'x_center': 133, 'y_center': 194, 'rotation_angle': -30, 'luminosity_threshold': 42000},
   'e09': {'x_center': 113, 'y_center': 203, 'rotation_angle': -13, 'luminosity_threshold': 42000},

   'f01': {'x_center': 229, 'y_center': 204, 'rotation_angle':  25, 'luminosity_threshold': 42000},
   'f02': {'x_center': 195, 'y_center': 180, 'rotation_angle':  54, 'luminosity_threshold': 43000},
   'f03': {'x_center': 181, 'y_center': 155, 'rotation_angle':  70, 'luminosity_threshold': 42000},
   'f04': {'x_center': 188, 'y_center':  86, 'rotation_angle': -58, 'luminosity_threshold': 40000},
   'f05': {'x_center': 206, 'y_center':  65, 'rotation_angle': -35, 'luminosity_threshold': 42000},
   'f06': {'x_center': 236, 'y_center':  49, 'rotation_angle': -20, 'luminosity_threshold': 41000},

   'g01': {'x_center':  34, 'y_center':  62, 'rotation_angle': -60, 'luminosity_threshold': 42000},
   'g02': {'x_center':  47, 'y_center':  44, 'rotation_angle': -44, 'luminosity_threshold': 42000},
   'g03': {'x_center':  78, 'y_center':  26, 'rotation_angle': -14, 'luminosity_threshold': 42000},
   'g04': {'x_center':  98, 'y_center':  25, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'g05': {'x_center': 124, 'y_center':  25, 'rotation_angle':   0, 'luminosity_threshold': 39500},
   'g06': {'x_center': 150, 'y_center':  26, 'rotation_angle':   0, 'luminosity_threshold': 41000},
   'g07': {'x_center': 214, 'y_center':  29, 'rotation_angle':   0, 'luminosity_threshold': 39500},
   'g08': {'x_center': 246, 'y_center':  28, 'rotation_angle':   0, 'luminosity_threshold': 40000},
   'g09': {'x_center': 281, 'y_center':  32, 'rotation_angle':  17, 'luminosity_threshold': 42000},
   'g10': {'x_center': 312, 'y_center':  47, 'rotation_angle':  38, 'luminosity_threshold': 42000},
   'g11': {'x_center': 328, 'y_center':  65, 'rotation_angle':  58, 'luminosity_threshold': 42000},

   'h01': {'x_center': 322, 'y_center': 197, 'rotation_angle': -53, 'luminosity_threshold': 45000},
   'h02': {'x_center': 300, 'y_center': 214, 'rotation_angle': -31, 'luminosity_threshold': 43000},
   'h03': {'x_center': 278, 'y_center': 224, 'rotation_angle': -14, 'luminosity_threshold': 42000},
   'h04': {'x_center': 261, 'y_center': 226, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'h05': {'x_center': 232, 'y_center': 225, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'h06': {'x_center': 204, 'y_center': 225, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'h07': {'x_center': 177, 'y_center': 223, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'h08': {'x_center': 145, 'y_center': 223, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'h09': {'x_center': 114, 'y_center': 223, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'h10': {'x_center':  85, 'y_center': 222, 'rotation_angle':   7, 'luminosity_threshold': 42000},
   'h11': {'x_center':  60, 'y_center': 213, 'rotation_angle':  26, 'luminosity_threshold': 42000},
   'h12': {'x_center':  45, 'y_center': 205, 'rotation_angle':  42, 'luminosity_threshold': 42000},
   'h13': {'x_center':  30, 'y_center': 183, 'rotation_angle':  63, 'luminosity_threshold': 42000},

   'i01': {'x_center': 114, 'y_center': 23, 'rotation_angle':  -10, 'luminosity_threshold': 42000},
   'i02': {'x_center': 139, 'y_center': 16, 'rotation_angle':  -19, 'luminosity_threshold': 40500},
   'i03': {'x_center': 164, 'y_center': 12, 'rotation_angle':    0, 'luminosity_threshold': 42000},
   'i04': {'x_center': 181, 'y_center': 13, 'rotation_angle':    0, 'luminosity_threshold': 41000},
   'i05': {'x_center': 198, 'y_center': 13, 'rotation_angle':    6, 'luminosity_threshold': 42000},
   'i06': {'x_center': 218, 'y_center': 18, 'rotation_angle':   22, 'luminosity_threshold': 39500},
   'i07': {'x_center': 237, 'y_center': 25, 'rotation_angle':   20, 'luminosity_threshold': 40000},

   'j01': {'x_center': 225, 'y_center':  13, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'j02': {'x_center': 250, 'y_center':  13, 'rotation_angle':   0, 'luminosity_threshold': 42000},
   'j03': {'x_center': 278, 'y_center':  16, 'rotation_angle':  12, 'luminosity_threshold': 42000},
   'j04': {'x_center': 301, 'y_center':  24, 'rotation_angle':  26, 'luminosity_threshold': 40000},
   'j05': {'x_center': 332, 'y_center':  38, 'rotation_angle':  25, 'luminosity_threshold': 42000},
   'j06': {'x_center': 364, 'y_center':  54, 'rotation_angle':  25, 'luminosity_threshold': 42000},
   'j07': {'x_center': 390, 'y_center':  76, 'rotation_angle':  49, 'luminosity_threshold': 42000},
   'j08': {'x_center': 404, 'y_center': 107, 'rotation_angle':  81, 'luminosity_threshold': 43000},
   'j09': {'x_center': 405, 'y_center': 125, 'rotation_angle':  90, 'luminosity_threshold': 43000},
   'j10': {'x_center': 403, 'y_center': 160, 'rotation_angle': -73, 'luminosity_threshold': 42000},
   'j11': {'x_center': 395, 'y_center': 176, 'rotation_angle': -59, 'luminosity_threshold': 43000},
   'j12': {'x_center': 370, 'y_center': 201, 'rotation_angle': -30, 'luminosity_threshold': 43000},
   'j13': {'x_center': 337, 'y_center': 216, 'rotation_angle': -23, 'luminosity_threshold': 45000},
   'j14': {'x_center': 313, 'y_center': 224, 'rotation_angle':  -9, 'luminosity_threshold': 45000},
   'j15': {'x_center': 287, 'y_center': 226, 'rotation_angle':   0, 'luminosity_threshold': 43000}
  }