#!/usr/bin/python

# sub-parts of the U-Net model
''' Make necessary imports'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    ''' 
    Constrct a Convolution Block
    consisting of Two Conv2D layers each with Batch-Normalisation and
    ReLU layer applied consecutively.
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__() # way to call super() in Python2 now redundant.
        '''
        Creates a Sequential object 
        which represents one convolutional block.

        parameters
            in_ch : Number of channels in the input image
            out_ch : Number of channels produced by the convolution
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ''' Returns the Sequential Object. '''
        x = self.conv(x)
        return x


class inconv(nn.Module):
    '''
    Construct the Convolution Block (CB)
    using the double_conv class.
    Is the first CB that handles the input.
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        ''' 
        Create object of double_conv class
        
        parameters
            in_ch : Number of channels in the input image
            out_ch : Number of channels produced by the convolution
        '''
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        ''' Returns the convolution block. '''
        x = self.conv(x)
        return x


class down(nn.Module):
    '''
    Create an Encoder block,
    consists of a Max-pooling layer and
    a Convolution Block
    '''
    def __init__(self, in_ch, out_ch):
        '''
        Creates a Sequential object 
        which represents one Encoder block.

        parameters
            in_ch : Number of channels in the input image
            out_ch : Number of channels produced by the convolution
        '''
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        ''' Return the encoder block that performs downsampling.'''
        x = self.mpconv(x)
        return x


class up(nn.Module):
    '''
    Create a Decoder block,
    consists of a UpSampling operation and
    a Convolution Block
    '''
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        ''' Return the decoder block that performs downsampling.'''
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    '''
    Construct a Conv2D layer.
    Is the last CB that produces the output.
    '''
    def __init__(self, in_ch, out_ch):
        ''' 
        Create object Conv2D
        
        parameters
            in_ch : Number of channels in the input image
            out_ch : Number of channels produced by the convolution
        '''
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        ''' Return the layer. '''
        x = self.conv(x)
        return x
