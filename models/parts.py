import torch
import torch.nn as nn
import torch.nn.functional as F


def DoubleConv(in_channel, out_channel):
	conv = nn.Sequential(
		nn.Conv2d(in_channel, out_channel, kernel_size = 3),
		nn.ReLU(inplace = True),
		nn.Conv2d(out_channel, out_channel, kernel_size = 3),
		nn.ReLU(inplace = True)
	)
	return conv

def crop(original, target):
	target_size = target.size()[2]
	original_size = original.size()[2]
	delta = original_size - target_size
	delta = delta //2
	return original[:, :, delta:original_size - delta, delta:original_size - delta]

