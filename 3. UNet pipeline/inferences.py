''' Make necessary imports'''
from tqdm import tqdm
import os
import torch
from build_model import *
from torch.utils.data import DataLoader
from data_loader import LungSegVal
from torchvision import transforms
import torch.nn.functional as F    
import numpy as np
from skimage import morphology, color, io, exposure

''' Make visible GPU device with UUID 0 to this CUDA Application'''
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape[:2]
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3))^gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    
    img_hsv = color.rgb2hsv(img)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':
    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    # Load test data
    img_size = (864, 864)
    inp_shape = (864,864,3)
    batch_size=1

    # Load model
    net = UNet(3,1)
    ''' Move model and its parameters to cuda '''
    net.cuda()

    net.load_state_dict(torch.load('Weights_BCE_Dice_InvDice/cp_bce_flip_lr_04_no_rot52_0.04634043872356415.pth.tar'))
    ''' Set model in evaluating mode '''
    net.eval()

    ''' To apply Transfomations to Test images '''  
    transformations_test = transforms.Compose([transforms.Resize((864,864)),transforms.ToTensor()]) 
    ''' Load and Prepare Testing data '''
    test_set = LungSegVal(transforms = transformations_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle = False)
    ious = np.zeros(len(test_loader))
    dices = np.zeros(len(test_loader))
    if not(os.path.exists('./results_Unet_BCE_Dice_InvDice_test')):
        os.mkdir('./results_Unet_BCE_Dice_InvDice_test')
    
    i = 0
    for xx, yy, name in tqdm(test_loader):
        ''' Move image cuda '''
        xx = xx.cuda()
        yy = yy
        
        name = name[0][:-4]
        print (name)

        ''' Obtain a prediction'''
        pred = net(xx)
        pred = F.sigmoid(pred)
        ''' Move pred tensor from GPU to CPU'''
        pred = pred.cpu()
        ''' Detahc tensor from tracking history, and prevent future computation from being tracked.'''
        pred = pred.detach().numpy()[0,0,:,:]
        mask = yy.numpy()[0,0,:,:]
        xx = xx.cpu()
        xx = xx.numpy()[0,:,:,:].transpose(1,2,0)
        ''' Return image after stretching or shrinking its intensity levels to range between 0 and 1'''
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))

        # Binarize masks
        gt = mask > 0.5
        pr = pred > 0.5

        # Remove regions smaller than 2% of the image
        #pr = remove_small_regions(pr, 0.02 * np.prod(img_size))
       
        ''' Save the predictions. '''
        io.imsave('results_Unet_BCE_Dice_InvDice_test/{}.png'.format(name), pr*255)

        ''' Calculate metrics IOU and Dice'''
        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        
        i += 1
        if i == len(test_loader):
            break

    print ('Mean IoU:', ious.mean())
    print ('Mean Dice:', dices.mean())
    
    
    
