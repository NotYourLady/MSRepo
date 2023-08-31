import torch
import torch.nn as nn


class swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)
    

class GLU(nn.Module):
    """Gated Linear Unit"""
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class norm_act(nn.Module):
    def __init__(self, channels, act=swish()):
        super(norm_act, self).__init__()
        self.act = act
        #self.norm = #nn.InstanceNorm3d(channels, affine=False)
        #self.norm = torch.nn.LazyInstanceNorm3d()
        #self.norm = torch.nn.BatchNorm3d(channels)
        self.norm = nn.Identity(channels)
        
    def forward(self, x):
        normed_x = self.norm(x)
        out = self.act(normed_x)
        return out