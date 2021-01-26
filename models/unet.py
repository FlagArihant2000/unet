import torch.nn.functional as F
from .parts import *

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.down_conv_1 = DoubleConv(3, 64)
    self.down_conv_2 = DoubleConv(64, 128)
    self.down_conv_3 = DoubleConv(128, 256)
    self.down_conv_4 = DoubleConv(256, 512)
    self.down_conv_5 = DoubleConv(512, 1024)

    self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
    self.up_conv_1 = DoubleConv(1024, 512)

    self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
    self.up_conv_2 = DoubleConv(512, 256)

    self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
    self.up_conv_3 = DoubleConv(256, 128)

    self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
    self.up_conv_4 = DoubleConv(128, 64)

    self.out = nn.Conv2d(64, 32, kernel_size = 1)

  
  def forward(self, image):
    # Batch size, channel, height, width
    # Encoder
    x1 = self.down_conv_1(image)
    x2 = self.max_pool(x1)

    x3 = self.down_conv_2(x2)
    x4 = self.max_pool(x3)

    x5 = self.down_conv_3(x4)
    x6 = self.max_pool(x5)

    x7 = self.down_conv_4(x6)
    x8 = self.max_pool(x7)

    x9 = self.down_conv_5(x8)

    # Decoder
    x = self.up_trans_1(x9)
    y = crop(x7, x)
    x = self.up_conv_1(torch.cat([x, y], 1))

    x = self.up_trans_2(x)
    y = crop(x5, x)
    x = self.up_conv_2(torch.cat([x, y], 1))

    x = self.up_trans_3(x)
    y = crop(x3, x)
    x = self.up_conv_3(torch.cat([x, y], 1))

    x = self.up_trans_4(x)
    y = crop(x1, x)
    x = self.up_conv_4(torch.cat([x, y], 1))

    x = self.out(x)
    return x
