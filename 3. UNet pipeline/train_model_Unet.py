''' Imports '''
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from build_model import *
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

''' Make visible GPU device with UUID 0 to this CUDA Application'''
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Average(object):
    '''
    Define Average class 
    to calculate a running average of the Loss used during training.
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    def avg(self):
        return self.sum / self.count
#------------------------------        
# import csv
''' Write Summaries and Events to an Event file '''
writer = SummaryWriter()
#----------------------------------------
class SoftDiceLoss(nn.Module):
    '''
    Define Soft Dice Loss
    '''        
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))


            
class InvSoftDiceLoss(nn.Module):
    '''
    Define Inverted Soft Dice Loss
    '''   
    def __init__(self, weight=None, size_average=True):
        super(InvSoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()
    
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))


#Tranformations------------------------------------------------

''' Apply Transfomations to Training images '''
transformations_train = transforms.Compose([transforms.Resize((1024,1024)),transforms.ToTensor()])

''' Apply Transfomations to Validation images '''    
transformations_val = transforms.Compose([transforms.Resize((1024,1024)),transforms.ToTensor()])     
#-------------------------------------------------------------                                      

''' Imports from other modules'''      
from data_loader import LungSegTrain
from data_loader import LungSegVal

train_set = LungSegTrain(transforms = transformations_train)

''' Training Hyperparameters '''
batch_size = 1   
num_epochs = 75
    
def train():
    '''
    Function to train the Segmentation Model
    '''
    cuda = torch.cuda.is_available()
    net = UNet(3,1)
    if cuda:
        ''' Move model and its parameters to cuda '''
        net = net.cuda()
    #net.load_state_dict(torch.load('Weights_BCE_Dice/cp_bce_lr_05_100_0.222594484687.pth.tar'))
    ''' Move the criterion to cuda '''
    criterion1 = nn.BCEWithLogitsLoss().cuda()
    criterion2 = SoftDiceLoss().cuda()
    criterion3 = InvSoftDiceLoss().cuda()
    #criterion4 = W_bce().cuda()
    #criterion5 = int_custom_loss()
    #criterion6 = weighted_dice_invdice()     
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    #scheduler = MultiStepLR(optimizer, milestones=[2,10,75,100], gamma=0.1)
    
    ''' Prepare Training data '''
    print("preparing training data ...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("done ...")

    ''' Load and Prepare Validation data '''
    val_set = LungSegVal(transforms = transformations_val)   
    val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=False)

    ''' Begin Training '''
    for epoch in tqdm(range(num_epochs)):
        #scheduler.step()        
        train_loss = Average()
        ''' Set model in training mode '''
        net.train()
        for i, (images, masks) in tqdm(enumerate(train_loader)):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            ''' Clear x.grad for every parameter x in optimizer to prevent accumulation of gradients'''
            optimizer.zero_grad()
            ''' Obtain a prediction '''
            outputs = net(images)
            #writer.add_image('Training Input',images)
            #writer.add_image('Training Pred',F.sigmoid(outputs)>0.5)
            ''' Calculate a combined loss '''
            c1 = criterion1(outputs,masks) + criterion2(outputs, masks) + criterion3(outputs, masks)
            loss = c1
            writer.add_scalar('Train Loss',loss,epoch)
            ''' Computes Gradient (dloss/dx) of Loss wrt Trainable parameters x '''
            loss.backward()
            ''' Updat eth value of x using gradient x.grad  '''
            optimizer.step()
            ''' Calculate a running average of the comined loss '''
            train_loss.update(loss.item(), images.size(0))

            for param_group in optimizer.param_groups:
	            writer.add_scalar('Learning Rate',param_group['lr'])

        ''' Create objects for calculating current average'''
        val_loss1 = Average()
        val_loss2 = Average()
        val_loss3 = Average()
        ''' Set model in evaluating mode '''
        net.eval()

        ''' Perform Validation '''
        for images, masks,_ in tqdm(val_loader):
            images = Variable(images)
            masks = Variable(masks)
            ''' Move image and mask to cuda'''
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            ''' Make predictions '''
            outputs = net(images)

            ''' Capture into event file at every tenth epoch '''
            if (epoch)%10==0:
                writer.add_image('Validation Input',images,epoch)
                writer.add_image('Validation GT ',masks,epoch)
                writer.add_image('Validation Pred0.5',F.sigmoid(outputs)>0.5,epoch)
                writer.add_image('Validation Pred0.3',F.sigmoid(outputs)>0.3,epoch)
                writer.add_image('Validation Pred0.65',F.sigmoid(outputs)>0.65,epoch)
            
            ''' Calcuate Validation Losses and capture '''
            vloss1 = criterion1(outputs, masks)
            vloss2 = criterion2(outputs, masks)  
            vloss3 = criterion3(outputs, masks) #+ criterion2(outputs, masks)
            #vloss = vloss2 + vloss3
            writer.add_scalar('Validation loss(BCE)',vloss1,epoch)
            writer.add_scalar('Validation loss(Dice)',vloss2,epoch)
            writer.add_scalar('Validation loss(InvDice)',vloss3,epoch)

            ''' Update Average Losses '''
            val_loss1.update(vloss1.item(), images.size(0))
            val_loss2.update(vloss2.item(), images.size(0))
            val_loss3.update(vloss3.item(), images.size(0))

        print("Epoch {}, Training Loss(BCE+Dice): {}, Validation Loss(BCE): {}, Validation Loss(Dice): {}, Validation Loss(InvDice): {}".format(epoch+1, train_loss.avg(), val_loss1.avg(), val_loss2.avg(), val_loss3.avg()))

        # with open('Log.csv', 'a') as logFile:
        #     FileWriter = csv.writer(logFile)
        #     FileWriter.writerow([epoch+1, train_loss.avg, val_loss1.avg, val_loss2.avg, val_loss3.avg])        
        	
        ''' Save the state of the Network at every epoch into disk file '''
        torch.save(net.state_dict(), 'Weights_BCE_Dice_InvDice/cp_bce_flip_lr_04_no_rot{}_{}.pth.tar'.format(epoch+1, val_loss2.avg()))                
    return net

def test(model):
    ''' 
    Define a Test Mode for the model.
    For this, set model to evaluation mode.
    '''
    model.eval()

if __name__ == "__main__":
    train()
