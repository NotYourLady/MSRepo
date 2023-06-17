import torch
import torch.nn as nn
import torch.nn.functional as F

class IOU_Metric():
    def __init__(self, eps=1e-5, thresh=None):
        self.eps = eps
        self.thresh = thresh

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W x D => BATCH x H x W x D
        labels = labels.squeeze(1).byte()
        
        SMOOTH = 1e-8
        intersection = (outputs & labels).float().sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2, 3))         # Will be zzero if both are 0

        iou = (intersection + self.eps) / (union + self.eps)
        if self.thresh is not None:
            threshold = self.thresh
            thresholded = torch.clamp(iou - self.thresh, 0, 1).ceil()  # This is equal to comparing with thresolds
            return thresholded  #
        return iou   




# class DiceMetric(nn.Module):
#     def __init__(self, eps=1e-5):
#         super(Dice_metric, self).__init__()
#         self.eps = eps

#     def forward(self, inputs, targets, logits=True):
#         categories = inputs.shape[1]
#         targets = targets.contiguous()
#         targets = one_hot(targets, categories)
#         if logits:
#             inputs = torch.argmax(F.softmax(inputs, dim=1), dim=1)
#         inputs = one_hot(inputs, categories)

#         dims = tuple(range(2, targets.ndimension()))
#         tps = torch.sum(inputs * targets, dims)
#         fps = torch.sum(inputs * (1 - targets), dims)
#         fns = torch.sum((1 - inputs) * targets, dims)
#         loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
#         return loss[:, 1:].mean(dim=1)    


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
    

class DiceLoss:
    def __init__(self, discrepancy=1):
        self.discrepancy = discrepancy

    def __call__(self, y_real, y_pred):
        num = 2*torch.sum(y_real*y_pred) + self.discrepancy
        den = torch.sum(y_real + y_pred) + self.discrepancy
        res = 1 - (num/den)
        return res 


class WeightedExpBCE:
    def __init__(self, gamma_bce, bce_weight=1, eps=1e-8):
        self.bce_weight = bce_weight
        self.gamma_bce = gamma_bce
        self.eps = 1e-8

    def set_bce_weight(self, freq):
        assert freq > 0
        assert freq < 1
        w1 = (1 / freq) ** 0.5
        w2 = (1 / (1 - freq) ) ** 0.5
        self.bce_weight = w1 / w2
    
    def __call__(self, y_real, y_pred):
        first_term = torch.pow(- y_real * torch.log(y_pred + self.eps) + self.eps, self.gamma_bce)
        second_term = - (1 - y_real) * torch.log(1 - y_pred + self.eps)
        second_term = torch.pow(second_term + self.eps, self.gamma_bce)
        return torch.mean(self.bce_weight * first_term + second_term)
    

class ExponentialLogarithmicLoss:
    def __init__(self, gamma_tversky = None, gamma_bce = None,
                 freq=None, lamb=0.5, tversky_alfa=0.5):
        assert gamma_tversky is not None
        assert freq is not None
        if gamma_bce is None:
            gamma_bce = gamma_tversky
        self.lamb = lamb
        self.weighted_bce_loss = WeightedExpBCE(gamma_bce)
        self.weighted_bce_loss.set_bce_weight(freq)
        self.gamma_tversky = gamma_tversky
        self.tversky = TverskyLoss(tversky_alfa)
        self.eps = 1e-8
        
        
    def __call__(self, y_real, y_pred):
        w_exp_bce = self.weighted_bce_loss(y_real, y_pred)
        log_tversky = -torch.log(1-self.tversky(y_real, y_pred) + self.eps)
        epx_log_tversky = torch.pow(log_tversky + self.eps, self.gamma_tversky)
        #print("w_exp_bce:", w_exp_bce, "epx_log_tversky:", epx_log_tversky)
        return self.lamb * w_exp_bce + (1 - self.lamb) * epx_log_tversky

    
class SumLoss:
    def __init__(self, alfa=0.5):
        self.eps = 1e-3
        self.alfa = alfa
        
    def __call__(self, y_real, y_pred):
        k = y_real.sum()/y_pred.sum()
        k = torch.clip(k, min=0.1, max=10)
        #print("k:", k)
        loss = torch.log(self.alfa * torch.exp(k) + (1-self.alfa) * torch.exp(1/(k+self.eps)))-1
        return loss
    

class LinearCombLoss:
    def __init__(self, funcs_and_сoef_list):
        self.funcs_and_сoef_list = funcs_and_сoef_list
        
    def __call__(self, y_real, y_pred):
        loss = 0
        for func, coef in self.funcs_and_сoef_list:
            f_i = func(y_real, y_pred)
            #print("f_i:", f_i)
            loss += coef * func(y_real, y_pred)
            
        return loss
    
    
class MultyscaleLoss(nn.Module):
    def __init__(self, loss):
        super(MultyscaleLoss, self).__init__()
        self.loss = loss
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, y_real, outs):
        out_loss = 0
        for idx, out in enumerate(outs):
            #print("y_real:", y_real.shape, "out:",  out.shape)
            #out_loss += 2**(-3 * idx) * self.loss(y_real, out)
            out_loss += 2**(-idx) * self.loss(y_real, out)
            y_real = self.pool(y_real)
            return(out_loss) ### TEST
    
        return(out_loss)    