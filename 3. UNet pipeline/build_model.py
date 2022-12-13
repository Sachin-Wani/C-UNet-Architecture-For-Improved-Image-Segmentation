''' Make necessary imports'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from unet_parts import *

class UNet(nn.Module):
    '''
    Class to construct the UNet model
    inherits nn.Module
    '''
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        ''' handle input '''
        self.inc = inconv(n_channels, 64)
        ''' Four Encoder Blocks '''
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        ''' Four Decoder Blocks '''
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        ''' Final Convolution layer '''
        self.outc = outconv(64, n_classes)
        self.Dropout = nn.Dropout(0.5)

    def forward(self, x):
        ''' Apply the operations '''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.Dropout(x3)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
