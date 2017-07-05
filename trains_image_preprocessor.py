#!/usr/bin/env python

'''
cut selection: start: 0x131;
size: 960x480    944x480

my_photo-2187
train
340x102 (10x40)

image-1531.jpg
340x105; (10x40)

def echo(*args):
  """Print the arguments on standard output"""
  print "echo:", args

register(
  "console_echo", "", "", "", "", "",
  "<Toolbox>/Xtns/Languages/Python-Fu/Test/_Console Echo", "",
  [
  (PF_STRING, "arg0", "argument 0", "test string"),
  (PF_INT,    "arg1", "argument 1", 100          ),
  (PF_FLOAT,  "arg2", "argument 2", 1.2          ),
  (PF_COLOR,  "arg3", "argument 3", (0, 0, 0)    ),
  ],
  [],
  echo
  )

main()


from gimpfu import *

def python_clothify(timg, tdrawable, bx=9, by=9,
                    azimuth=135, elevation=45, depth=3):
    width = tdrawable.width
    height = tdrawable.height

    img = gimp.Image(width, height, RGB)
    img.disable_undo()

    layer_one = gimp.Layer(img, "X Dots", width, height, RGB_IMAGE,
                           100, NORMAL_MODE)
    img.add_layer(layer_one, 0)
    pdb.gimp_edit_fill(layer_one, BACKGROUND_FILL)

    pdb.plug_in_noisify(img, layer_one, 0, 0.7, 0.7, 0.7, 0.7)

    layer_two = layer_one.copy()
    layer_two.mode = MULTIPLY_MODE
    layer_two.name = "Y Dots"
    img.add_layer(layer_two, 0)

    pdb.plug_in_gauss_rle(img, layer_one, bx, 1, 0)
    pdb.plug_in_gauss_rle(img, layer_two, by, 0, 1)

    img.flatten()

    bump_layer = img.active_layer

    pdb.plug_in_c_astretch(img, bump_layer)
    pdb.plug_in_noisify(img, bump_layer, 0, 0.2, 0.2, 0.2, 0.2)
    pdb.plug_in_bump_map(img, tdrawable, bump_layer, azimuth,
                         elevation, depth, 0, 0, 0, 0, True, False, 0)

    gimp.delete(img)

register(
        "python_fu_clothify",
        "Make the specified layer look like it is printed on cloth",
        "Make the specified layer look like it is printed on cloth",
        "James Henstridge",
        "James Henstridge",
        "1997-1999",
        "<Image>/Filters/Artistic/_Clothify...",
        "RGB*, GRAY*",
        [
                (PF_INT, "x_blur", "X blur", 9),
                (PF_INT, "y_blur", "Y blur", 9),
                (PF_INT, "azimuth", "Azimuth", 135),
                (PF_INT, "elevation", "Elevation", 45),
                (PF_INT, "depth", "Depth", 3)
        ],
        [],
        python_clothify)

main()



(define (resize-image-keep-ratio filename-in filename-out new-width new-height)
  (let* ((image      (car (gimp-file-load RUN-NONINTERACTIVE filename-in "")))
         (drawable   (car (gimp-image-active-drawable image)))
         (cur-width  (car (gimp-image-width image)))
         (cur-height (car (gimp-image-height image)))
         (ratio      (min (/ new-width cur-width) (/ new-height cur-height)))
         (width      (* ratio cur-width))
         (height     (* ratio cur-height))
        )

     (gimp-image-scale image width height)
     (gimp-file-save   RUN-NONINTERACTIVE image drawable filename-out "")
  )
)

(define (simple-unsharp-mask filename
                              radius
                  amount
                  threshold)
   (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
          (drawable (car (gimp-image-get-active-layer image))))
     (plug-in-unsharp-mask RUN-NONINTERACTIVE
                       image drawable radius amount threshold)
     (gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)
     (gimp-image-delete image)))


gimp-image-convert-grayscale image
gimp-image-crop image new-width new-height offx offy


gimp --no-interface --batch '(python-fu-console-echo RUN-NONINTERACTIVE "another string" 777 3.1416 (list 1 0 0))' '(gimp-quit 1)'

c:\> start gimp-2.2.exe -d -i -b  "(resize-image-keep-ratio \"c:\\temp\\ex_10.jpg\" \"c:\\temp\\ex_10_resized.jpg\" 100 100)" "(gimp-quit 0)"

https://www.gimp.org/tutorials/Basic_Batch/
gimp -i -b '(batch-unsharp-mask "*.png" 5.0 0.5 0)' -b '(gimp-quit 0)'



  (define (batch-unsharp-mask pattern
                              radius
                              amount
                              threshold)
  (let* ((filelist (cadr (file-glob pattern 1))))
    (while (not (null? filelist))
           (let* ((filename (car filelist))
                  (image (car (gimp-file-load RUN-NONINTERACTIVE
                                              filename filename)))
                  (drawable (car (gimp-image-get-active-layer image))))
             (plug-in-unsharp-mask RUN-NONINTERACTIVE
                                   image drawable radius amount threshold)
             (gimp-file-save RUN-NONINTERACTIVE
                             image drawable filename filename)
             (gimp-image-delete image))
           (set! filelist (cdr filelist)))))




http://matthiaseisen.com/pp/patterns/p0202/ Crop images with PIL/Pillow
https://stackoverflow.com/questions/24745857/python-pillow-how-to-scale-an-image
https://stackoverflow.com/questions/23935840/converting-an-rgb-image-to-grayscale-and-manipulating-the-pixel-data-in-python


'''

""" 
chmod +x trains_image_preprocessor.py
cp trains_image_preprocessor.py ~/.gimp-2.8/plug-ins/
gimp --no-interface --batch '(python-fu-trains-image-preprocessor RUN-NONINTERACTIVE)' '(gimp-quit 1)'

devel-docs/libgimp/html/libgimp.html
"""

import os
from gimpfu import *
import glob

def trains_image_preprocessor():
  filelist = sorted(glob.glob('./trains_images_dataset/*.jpg'))
  for filename in filelist:
    print "Filename: %s" % filename
    image = pdb.gimp_file_load(filename, filename, run_mode=RUN_NONINTERACTIVE)
    pdb.gimp_image_convert_grayscale(image)
    pdb.gimp_image_crop(image, 960, 480, 0, 130)
    pdb.gimp_context_set_interpolation(INTERPOLATION_LANCZOS)
    pdb.gimp_image_scale(image, 480, 240)
    drawable = pdb.gimp_image_get_active_layer(image)
    output_file = './trains_images_dataset/preprocessed/' + os.path.basename(filename)
    pdb.gimp_file_save(image, drawable, output_file, output_file, run_mode=RUN_NONINTERACTIVE)
    pdb.gimp_image_delete(image)

register("trains_image_preprocessor", "", "", "", "", "", "", "",
         [], [], trains_image_preprocessor)

main()
