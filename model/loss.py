import torch
import torch.nn as nn
import torch.nn.functional as F
balance_cof = 4


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    # focal_inds = pred.gt(1.4013e-45) * pred.lt(1-1.4013e-45)
    pred = torch.clamp(pred, 1.4013e-45, 1)
    fos = torch.sum(gt, 1)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    fos_inds = torch.unsqueeze(fos.gt(0).float(), 1)
    fos_inds = fos_inds.expand(-1, 3, -1, -1, -1)
    neg_inds = neg_inds + (balance_cof - 1) * fos_inds * gt.eq(0)

    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    # pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * focal_inds
    # neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * focal_inds
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * balance_cof
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def _MSE_loss(pred, gt):
    # mse_inds = pred.le(1.4013e-45) + pred.ge(1 - 1.4013e-45)
    pos_inds = gt.eq(1).float()
    num_pos = pos_inds.float().sum()
    mse_loss = torch.pow(pred - gt, 2)
    mse_loss = mse_loss.sum()

    if num_pos > 0:
        mse_loss = mse_loss / num_pos

    return mse_loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class FocalMSELoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalMSELoss, self).__init__()
        self.neg_loss = _neg_loss
        self.mse_loss = _MSE_loss

    def forward(self, out, target):
        return self.neg_loss(out, target) + self.mse_loss(out, target)


class _MSELoss_(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(_MSELoss_, self).__init__()
        self.mse_loss = _MSE_loss

    def forward(self, out, target):
        return self.mse_loss(out, target)


# class FocalLoss(nn.Module):
#
#     def __init__(self, focusing_param=2, balance_param=0.25):
#         super(FocalLoss, self).__init__()
#
#         self.focusing_param = focusing_param
#         self.balance_param = balance_param
#
#     def forward(self, output, target):
#         cross_entropy = F.cross_entropy(output, target)
#         cross_entropy_log = torch.log(cross_entropy)
#         logpt = - F.cross_entropy(output, target)
#         pt    = torch.exp(logpt)
#
#         focal_loss = -((1 - pt) ** self.focusing_param) * logpt
#
#         balanced_focal_loss = self.balance_param * focal_loss
#
#         return balanced_focal_loss
