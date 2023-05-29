import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(gt, categories):
    # Check the new function in PyTorch!!!
    size = [*gt.shape] + [categories]
    y = gt.view(-1, 1)
    gt = torch.FloatTensor(y.nelement(), categories).zero_().cuda()
    gt.scatter_(1, y, 1)
    gt = gt.view(size).permute(0, 4, 1, 2, 3).contiguous()
    return gt


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        n, ch, x, y, z = inputs.size()

        logpt = -self.criterion(inputs, targets.long())
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class tversky_loss(nn.Module):
    """
        Calculates the Tversky loss of the Foreground categories.
        if alpha == 1 --> Dice score
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
    """
    def __init__(self, alpha, eps=1e-5):
        super(tversky_loss, self).__init__()
        self.alpha = alpha
        self.beta = 2 - alpha
        self.eps = eps

    def forward(self, inputs, targets):
        # inputs.shape[1]: predicted categories
        targets = one_hot(targets, inputs.shape[1])
        inputs = F.softmax(inputs, dim=1)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims) * self.alpha
        fns = torch.sum((1 - inputs) * targets, dims) * self.beta
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        loss = torch.mean(loss, dim=0)
        return 1 - (loss[1:]).mean()


class segmentation_loss(nn.Module):
    def __init__(self, alpha):
        super(segmentation_loss, self).__init__()
        self.dice = tversky_loss(alpha=alpha, eps=1e-5)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        dice = self.dice(inputs, targets.contiguous())
        ce = self.ce(inputs, targets)
        return dice + ce


class Dice_metric(nn.Module):
    def __init__(self, eps=1e-5):
        super(Dice_metric, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, logits=True):
        categories = inputs.shape[1]
        targets = targets.contiguous()
        targets = one_hot(targets, categories)
        if logits:
            inputs = torch.argmax(F.softmax(inputs, dim=1), dim=1)
        inputs = one_hot(inputs, categories)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims)
        fns = torch.sum((1 - inputs) * targets, dims)
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        return loss[:, 1:].mean(dim=1)
    
    
def dice_loss(y_real, y_pred, discrepancy=1):
    num = 2*torch.sum(y_real*y_pred)
    den = torch.sum(y_real + y_pred) + discrepancy
    res = 1 - (num/den)
    return res 


def focal_loss(y_real, y_pred, eps = 1e-8, gamma = 0.5):
    first_term = torch.pow(1 - y_pred, gamma) * y_real * torch.log(y_pred + eps)
    #first_term = 0.5 * y_real * torch.log(y_pred + eps)
    second_term = (1 - y_real) * torch.log(1 - y_pred + eps)
    return( -torch.mean(first_term + second_term) )


class ComboLoss:
    def __init__(self, lamb = 0.5, gamma = 0.5):
        self.lamb = lamb
        self.gamma =  gamma

    def __call__(self, y_real, y_pred):
        focal = focal_loss(y_real, y_pred, eps = 1e-8, gamma = self.gamma)
        dice = dice_loss(y_real, y_pred, discrepancy=1.0e-6)
        #print("focal/dice", focal/dice)
        return self.lamb * focal + (1 - self.lamb) * dice

#TODO: протестировать усреднение bce до и после возведения в степень.
class ExponentialLogarithmicLoss:
    def __init__(self, gamma_dice = None, gamma_bce = None,
                       lamb=0.5, bce_weight=1):
            self.gamma_dice = gamma_dice
            self.gamma_bce =  gamma_bce
            self.lamb = lamb
            self.bce_weight = bce_weight
    
    def set_bce_weight(self, freq):
        w1 = (1 / freq) ** 0.5
        w2 = (1 / (1 - freq) ) ** 0.5
        self.bce_weight = w1 / w2
    
    def __call__(self, y_real, y_pred):
        w_exp_bce = self.weighted_bce_loss(y_real, y_pred)
        dice = torch.pow(self.log_dice_loss(y_real, y_pred), self.gamma_dice)
        print(w_exp_bce, dice)
        return self.lamb * w_exp_bce + (1 - self.lamb) * dice
    
    def log_dice_loss(self, y_real, y_pred, discrepancy=1, eps = 1e-8):
        num = 2*torch.sum(y_real * y_pred)
        den = torch.sum(y_real + y_pred) + discrepancy
        res = 1 - (num/den)
        print("res:", res)
        return -torch.log(res + eps)
    
    def weighted_bce_loss(self, y_real, y_pred, eps = 1e-8):
        first_term = torch.pow(-y_real * torch.log(y_pred + eps), self.gamma_bce)
        second_term = torch.pow(-(1-y_real) * torch.log(1 - y_pred + eps), self.gamma_bce)
        return(torch.mean(self.bce_weight * first_term + (1-self.bce_weight)*second_term) )
        #var2:
        #first_term = y_real * torch.log(y_pred + eps)
        #second_term = (1 - y_real) * torch.log(1 - y_pred + eps)
        #return( -torch.mean(first_term + second_term) )
            


    
    