import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from utils import *

class train(data.Dataset):
  def __init__(self, transform = None, imgdir = None, labeldir = None, transformlabel = None, colors = None):
    self.train_img = load_img(imgdir, gray = False, input = True)
    self.transform = transform
    self.transformlabel = transformlabel
    self.train_label = label(load_img(labeldir, gray = True, input = False), colors)
  def __len__(self):
    return len(self.train_img)
  def __getitem__(self, index):
    img = self.transform(self.train_img[index])
    label = self.transformlabel(self.train_label[index])
    return img, label

class validation(data.Dataset):
  def __init__(self, transform = None, imgdir = None, labeldir = None, transformlabel = None, colors = None):
    self.train_img = load_img(imgdir, gray = False, input = True)
    self.transform = transform
    self.transformlabel = transformlabel
    self.train_label = label(load_img(labeldir, gray = True, input = False), colors)
  def __len__(self):
    return len(self.train_img)
  def __getitem__(self, index):
    img = self.transform(self.train_img[index])
    label = self.transformlabel(self.train_label[index])
    return img, label

class test(data.Dataset):
  def __init__(self, transform = None, imgdir = None, labeldir = None, transformlabel = None, colors = None):
    self.train_img = load_img(imgdir, gray = False, input = True)
    self.transform = transform
    self.transformlabel = transformlabel
    self.train_label = label(load_img(labeldir, gray = True, input = False), colors)
  def __len__(self):
    return len(self.train_img)
  def __getitem__(self, index):
    img = self.transform(self.train_img[index])
    label = self.transformlabel(self.train_label[index])
    return img, label
