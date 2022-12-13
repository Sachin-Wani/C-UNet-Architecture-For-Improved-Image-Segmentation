''' Make necessary imports'''
import numpy as np
import os
import cv2
from skimage import io
from skimage.util import random_noise
from skimage.filters import gaussian
from tqdm import tqdm

image_names  = os.listdir("Images_padded1/Train/")

for name in tqdm(image_names):
    ''' Read images and masks '''
    img = io.imread('Images_padded1/Train/'+name)
    mask = io.imread('Mask_padded1/Train/'+name)

    ''' Apply random gaussian noise to images '''
    noise_image_01 = random_noise(img, mode='gaussian', seed=None, clip=True, var = 0.01)
    noise_image_01 = noise_image_01*255.
    noise_image_01 = noise_image_01.astype('uint8')

    ''' Save corresponding image and mask '''
    io.imsave('Images_padded1/Train/'+name[:-4]+'_01.png', noise_image_01)
    io.imsave('Mask_padded1/Train/'+name[:-4]+'_01.png', mask)

