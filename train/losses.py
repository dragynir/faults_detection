import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F




class DiceBCELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (inputs.sum() + outputs.sum() + smooth))
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


class FocalLoss(nn.Module):
     def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
    

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1 - bce_exp) ** gamma * bce

        return focal_loss



class TverskyLoss(nn.Module):
    '''
        https://arxiv.org/abs/1706.05721
        3D segmentation
    '''
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = ((1 - inputs) * targets).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1. - Tversky

