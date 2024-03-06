import torch
import torch.nn as nn

def get_total_params(model):
    total_params = sum(
    param.numel() for param in model.parameters()
    )
    return(total_params)


class DiceLoss:
    def __init__(self, discrepancy=1):
        self.discrepancy = discrepancy

    def __call__(self, y_real, y_pred):
        num = 2*torch.sum(y_real*y_pred) + self.discrepancy
        den = torch.sum(y_real + y_pred) + self.discrepancy
        res = 1 - (num/den)
        return res 

class TverskyLoss:
    def __init__(self, beta, discrepancy=0.5):
        self.discrepancy = discrepancy
        self.beta = beta

    def __call__(self, y_real, y_pred):   
        num = torch.sum(y_real*y_pred) + self.discrepancy
        den = num + self.beta * torch.sum( (1 - y_real) * y_pred) + \
              (1 - self.beta) * torch.sum(y_real * (1 - y_pred))
        res = 1 - (num/den)
        return res 

class DICE_Metric():
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W x D => BATCH x H x W x D
        labels = labels.squeeze(1).byte()
        
        TP = (outputs & labels).float().sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
        FN = ((1-outputs) & labels).float().sum((1, 2, 3))
        FP = (outputs & (1-labels)).float().sum((1, 2, 3))
        
        dice = (2*TP + self.eps) / (2*TP + FN + FP + self.eps)
        return dice