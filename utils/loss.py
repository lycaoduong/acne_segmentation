import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

class loss_function(nn.Module):
    """CrossEntropyLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 loss_space,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(loss_function, self).__init__()
        self.loss_space = loss_space
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.loss_space == 0:
            self.cls_criterion = dice_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

def dice_loss(pred,
             label,
             weight=None,
             reduction='mean',
             avg_factor=None,
             class_weight=None,
             ignore_index=None):

    loss = dice_loss_cal(pred, label)
    # loss2 = customBCELoss(pred, label)
    # loss = loss1 + loss2

    return loss

def dice_loss_cal(pred, target, smooth=1e-6):
    # comment out if your model contains a sigmoid or equivalent activation layer
    pred = torch.sigmoid(pred)
    loss = 0
    for i in range(pred.shape[1]):
        inputs = pred[:,i,:,:].flatten()
        targets = target[:,i,:,:].flatten()
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        loss_tem = 1 - dice
        loss += loss_tem

    loss  = loss/pred.shape[1]
    return loss

def IOU(pred, target, smooth=1e-6):
    pred = pred.sigmoid()
    pred = (pred>0.5).float()
    acc = 0
    for i in range(pred.shape[1]):
        overlap = pred[:,i,:,:].flatten() * target[:,i,:,:].flatten()
        union = pred[:,i,:,:].flatten() + target[:,i,:,:].flatten()
        # union = (union>=1).float()
        IOU = (overlap.sum()+smooth) / float(union.sum()+smooth)
        # correct = (pred[:,i,:,:].flatten() ==  target[:,i,:,:].flatten()).float().sum() / pred[:,i,:,:].flatten().shape[0]
        acc += IOU
    acc  = acc/pred.shape[1]
    # print(acc)
    return acc


def customBCELoss(pred, target):
    pred = torch.sigmoid(pred)
    loss = 0
    zeros = torch.zeros_like(target)
    if torch.cuda.is_available():
        zeros = zeros.cuda()
    bce = -(target * torch.log(pred) + (1. - target) * torch.log(1 - pred))
    # loss = torch.where(torch.ne(target, -1.0), bce, zeros)
    loss = bce.mean()
    return loss

class CustomBCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomBCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, eps=1e-4):
        zeros = torch.zeros_like(target)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        bce = -(target * torch.log(pred) + (1. - target) * torch.log(1 - pred))
        loss = torch.where(torch.ne(target, -1.0), bce, zeros)

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction =='mean':
            return loss.mean()
        else:
            return loss
