import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, prob_in, prob_gt, valid):
        loss = self.loss(prob_in, prob_gt) * valid
        return loss

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)

        # prevent NaN loss
        loss[torch.isnan(loss)] = 0
        return loss

class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid

        # prevent NaN loss
        loss[torch.isnan(loss)] = 0
        return loss

