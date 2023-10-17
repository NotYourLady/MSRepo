import torch
import torch.nn as nn
from itertools import combinations_with_replacement

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1,
                 padding_mode='reflect', act = nn.ReLU()):
        super().__init__()
        kernel_size_coef = (kernel_size-1)//2-1
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=dilation+kernel_size_coef, dilation=dilation,
                                 padding_mode=padding_mode)
        
        self.norm = torch.nn.InstanceNorm3d(out_channels, eps=1e-05,
                                            momentum=0.1, affine=False,
                                            track_running_stats=False)
        self.act = act
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MultuScaleDilationConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, stride=1, padding_mode='reflect'):
        super().__init__()
        
        self.Block1 = ConvBlock(in_channels, out_channels//4, kernel_size=1,
                                stride=stride, dilation=1, padding_mode=padding_mode)
        self.Block2 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=1, padding_mode=padding_mode)
        self.Block3 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=2, padding_mode=padding_mode)
        self.Block4 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=4, padding_mode=padding_mode)


    def forward(self, x):
        x1 = self.Block1(x)
        x2 = self.Block2(x)
        x3 = self.Block3(x)
        x4 = self.Block4(x)
        return torch.cat((x1, x2, x3, x4), 1)

    
class GaussianBlur3D(nn.Module):
    def __init__(self, sigma, device, fast=False, in_channels=1, kernel_size=7):
        super(GaussianBlur3D, self).__init__()
        self.device = device
        self.fast = fast
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        if self.fast:
            self.conv = nn.Conv3d(in_channels, in_channels, self.kernel_size, stride=1,
                                  padding=self.kernel_size//2, dilation=1, bias=False, 
                                  padding_mode='replicate').to(self.device)
        else:
            self.conv = nn.Conv3d(1, 1, self.kernel_size, stride=1, padding=self.kernel_size//2,
                                  dilation=1, bias=False, padding_mode='replicate').to(self.device)
        self.set_weights(sigma)
    
    def set_weights(self, sigma):
        assert sigma.shape in [torch.Size([1]), torch.Size([3])]
        kernel1d = torch.arange(self.kernel_size).type(torch.float32) - self.kernel_size//2
        kernel1d = kernel1d.to(self.device)
        if sigma.shape == torch.Size([1]):
            kernel1d = (kernel1d)**2 / 2 / sigma**2
            kernel1d = torch.exp(-kernel1d)
            kernel3d = torch.einsum('i,j,k->ijk', kernel1d, kernel1d, kernel1d)
        if sigma.shape == torch.Size([3]):
            kernel1 = (kernel1d)**2 / 2 / sigma[0]**2
            kernel2 = (kernel1d)**2 / 2 / sigma[1]**2
            kernel3 = (kernel1d)**2 / 2 / sigma[2]**2
            kernel1 = torch.exp(-kernel1)
            kernel2 = torch.exp(-kernel2)
            kernel3 = torch.exp(-kernel3)
            kernel3d = torch.einsum('i,j,k->ijk', kernel1, kernel2, kernel3)
        
        kernel3d /= kernel3d.sum()   
        if self.fast:
            kernel3d = torch.cat(self.in_channels*[kernel3d.unsqueeze(0),], axis=0)
            kernel3d = torch.cat(self.in_channels*[kernel3d.unsqueeze(0),], axis=0)
        else:
            kernel3d = kernel3d.unsqueeze(0).unsqueeze(0)
            self.conv.weight = torch.nn.Parameter(kernel3d, requires_grad=False)
        
    def forward(self, x, sigma=None):
        if sigma is not None:
            self.set_weights(sigma)    
        
        if self.fast:
            return(self.conv(x))
        else:
            outs = []
            channels = x.shape[1]
            for c in range(channels):
                outs.append(self.conv(x[:, c:c+1]))
            return(torch.cat(outs, axis=1))


class HessianTorch(nn.Module):
    def __init__(self, sigma, device, in_channels=1, eigvals_out=False):
        super(HessianTorch, self).__init__()
        self.eigvals_out = eigvals_out
        self.gauss = GaussianBlur3D(sigma=sigma, device=device,
                                         in_channels=in_channels, fast=True)
    
    def smart_norm(self, x):
        norms = torch.sum(x**2, dim=0).unsqueeze(0)
        normed = torch.nn.functional.normalize(x, p=2.0, dim=0, eps=1e-12)
        out = torch.cat([normed, norms], axis=0)
        return(out)
        
    
    def forward(self, vol, sigma):
        axes = [2, 3, 4]
        gaussian_filtered = self.gauss(vol, sigma)
        gradient = torch.gradient(gaussian_filtered, dim=axes)
        H_elems = [torch.gradient(gradient[ax0-2], axis=ax1)[0]
              for ax0, ax1 in combinations_with_replacement(axes, 2)]
        if self.eigvals_out:
            out = torch.stack(H_elems)
            return out
        else:
            out = torch.stack(H_elems)
            out = self.smart_norm(out)
            return out

class HessBlock(nn.Module):
    def __init__(self, start_scale, device, act=nn.Sigmoid(), in_channels=1, fc_channels=10):
        super(HessBlock, self).__init__()
        
        self.device = device
        start_scale = torch.tensor(start_scale*1, dtype=torch.float32, device=device)
        if start_scale.shape == torch.Size([]):
            start_scale = start_scale.unsqueeze(0)
        self.scale = nn.parameter.Parameter(data=start_scale)
        
        self.linear = nn.Sequential(
            nn.Linear(7+self.scale.shape[0], fc_channels, bias=True),
            nn.ReLU(),
            #nn.Linear(fc_channels, fc_channels, bias=True),
            #nn.ReLU(),
            nn.Linear(fc_channels, 1, bias=True),
            act
        )
        self.hess = HessianTorch(self.scale, device, in_channels=in_channels)
        self.flat = nn.Flatten(start_dim=2, end_dim=4)
        
        
    def forward(self, x):
        input_sizes = x.shape
        x = self.hess(x, self.scale).permute(1,2,3,4,5,0) #1
        x = self.flat(x)
        scale_attention = self.scale*torch.ones((x.shape[0], x.shape[1], x.shape[2],
                                                 self.scale.shape[0])).to(self.device)
        x = torch.cat([x, scale_attention], axis=3)
        x = self.linear(x) #2
        x = torch.unflatten(x, 2, input_sizes[2:])
        x = x.squeeze(-1)
        #print("HessBlock::squeeze:", x.shape)
        return x

    
class HessNet(nn.Module):
    def __init__(self, start_scale, device, out_act=nn.Sigmoid()):
        super(HessNet, self).__init__()
        
        self.device = device
        self.H1 = HessBlock(start_scale=[0.5], device=device, act=nn.ReLU())
        self.H2 = HessBlock(start_scale=[1.0], device=device, act=nn.ReLU())
        self.H3 = HessBlock(start_scale=[1.5], device=device, act=nn.ReLU())
        self.H4 = HessBlock(start_scale=[2.0], device=device, act=nn.ReLU())
        
        self.out_block = nn.Sequential(
            nn.Conv3d(in_channels=5, out_channels=1, kernel_size=5, stride=1,
                      padding=2, dilation=1, padding_mode="reflect"),
            out_act)

    def forward(self, x):
        h1 = self.H1(x)
        h2 = self.H2(x)
        h3 = self.H3(x)
        h4 = self.H4(x)
        x = torch.cat([x, h1, h2, h3, h4], 1)
        out = self.out_block(x)
        return(out)
    
    
    def to_device(self, device):
        self.to(device)
        
        self.H1.device = device
        self.H2.device = device
        self.H3.device = device
        self.H4.device = device
        
        self.H1.hess.gauss.device = device
        self.H2.hess.gauss.device = device
        self.H3.hess.gauss.device = device
        self.H4.hess.gauss.device = device
        
        
        
class HessNet2(nn.Module):
    def __init__(self, start_scale, device, channel_coef=4):
        super(HessNet2, self).__init__()
        
        
        self.device = device
        
        self.H1 = HessBlock(start_scale=start_scale, device=device,
                            act=nn.Sigmoid(), in_channels=3)
        self.H2 = HessBlock(start_scale=start_scale, device=device,
                            act=nn.Sigmoid(), in_channels=3)
        self.H3 = HessBlock(start_scale=start_scale, device=device,
                            act=nn.Sigmoid(), in_channels=6)
        self.H4 = HessBlock(start_scale=start_scale, device=device,
                            act=nn.Sigmoid(), in_channels=6)

        
        self.input_block = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=3),
            nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1,
                      padding=1, dilation=1, padding_mode="reflect"))

        self.inter_block = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=6, kernel_size=3, stride=1,
                      padding=1, dilation=1, padding_mode="reflect"),
            nn.ReLU())
    
        self.out_block = nn.Sequential(
            ConvBlock(in_channels=10, out_channels=5),
            nn.Conv3d(in_channels=5, out_channels=1, kernel_size=3, stride=1,
                      padding=1, dilation=1, padding_mode="reflect"),
            nn.Sigmoid())
    

    def forward(self, x_input):
        x_proc = self.input_block(x_input)
        h1 = self.H1(x_proc)
        h2 = self.H2(x_proc)
        H = torch.max(torch.cat([h1.unsqueeze(0),
                                 h2.unsqueeze(0)], axis=0), axis=0).values

        x = torch.cat([x_proc, H], 1) #6 ch
        x = self.inter_block(x) #6ch
        
        h3 = self.H3(x) 
        h4 = self.H4(x)
        H = torch.max(torch.cat([h3.unsqueeze(0),
                                 h4.unsqueeze(0)], axis=0), axis=0).values
        
        x = torch.cat([x_input, x_proc, H], 1) #10ch
        
        out = self.out_block(x)
        return(out)
    

    
class HessFeatures(nn.Module):
    def __init__(self, start_scale, device, out_act=nn.Sigmoid()):
        super(HessFeatures, self).__init__()
        
        self.device = device
        self.H1 = HessBlock(start_scale=[0.5], device=device, act=nn.ReLU())
        self.H2 = HessBlock(start_scale=[1.0], device=device, act=nn.ReLU())
        self.H3 = HessBlock(start_scale=[1.5], device=device, act=nn.ReLU())
        self.H4 = HessBlock(start_scale=[2.0], device=device, act=nn.ReLU())

    def forward(self, x):
        h1 = self.H1(x)
        h2 = self.H2(x)
        h3 = self.H3(x)
        h4 = self.H4(x)
        out = torch.cat([x, h1, h2, h3, h4], 1)
        return(out)
    
    
    
    def to_device(self, device):
        self.to(device)
        
        self.H1.device = device
        self.H2.device = device
        self.H3.device = device
        self.H4.device = device
        
        self.H1.hess.gauss.device = device
        self.H2.hess.gauss.device = device
        self.H3.hess.gauss.device = device
        self.H4.hess.gauss.device = device


class HessFeatures2(nn.Module):
    def __init__(self, start_scale, device, out_channels = 5, out_act=nn.Sigmoid()):
        super(HessFeatures2, self).__init__()
        
        self.device = device
        self.H1 = HessBlock(start_scale=[0.5], device=device, act=nn.ReLU())
        self.H2 = HessBlock(start_scale=[1.0], device=device, act=nn.ReLU())
        self.H3 = HessBlock(start_scale=[1.5], device=device, act=nn.ReLU())
        self.H4 = HessBlock(start_scale=[2.0], device=device, act=nn.ReLU())
        
        self.out_block = nn.Sequential(
            nn.Conv3d(in_channels=5, out_channels=out_channels, kernel_size=5, stride=1,
                      padding=2, dilation=1, padding_mode="reflect"),
            out_act)

    def forward(self, x):
        h1 = self.H1(x)
        h2 = self.H2(x)
        h3 = self.H3(x)
        h4 = self.H4(x)
        out = torch.cat([x, h1, h2, h3, h4], 1)
        out = self.out_block(out)
        return(out)
    
    
    
    def to_device(self, device):
        self.to(device)
        
        self.H1.device = device
        self.H2.device = device
        self.H3.device = device
        self.H4.device = device
        
        self.H1.hess.gauss.device = device
        self.H2.hess.gauss.device = device
        self.H3.hess.gauss.device = device
        self.H4.hess.gauss.device = device