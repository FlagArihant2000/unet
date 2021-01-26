import cv2
import numpy as np
import os

def load_img(directory, gray, input):
  images = []
  for img in os.listdir(directory):
    img = cv2.imread(os.path.join(directory, img))
    if input:
      img = cv2.resize(img, (572, 572), interpolation = cv2.INTER_AREA)
    else:
      img = cv2.resize(img, (388, 388), interpolation = cv2.INTER_AREA)
    if gray:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img)
  return images

def class_pix(labelimg, colors):
  class_pix = np.ones([388,388,1], dtype = int)
  for index, c in enumerate(colors):
    class_pix[labelimg == c] = index
  return class_pix

def label(imagelist, colors):
  images = []
  for img in imagelist:
    images.append(class_pix(img, colors))
  return images

def c2g(colors, grayscale):
  c2g = []
  for color in colors:
    color = np.reshape(color, (1,1,3))
    if grayscale:
      gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    else:
      gray = color
    c2g.append(gray)
  return c2g
