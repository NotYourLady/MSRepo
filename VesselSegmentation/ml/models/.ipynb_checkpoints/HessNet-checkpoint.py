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
                                            momentum=0.1, affine=False, track_running_stats=False)
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
    def __init__(self, sigma, device, in_channels=1):
        super(GaussianBlur3D, self).__init__()
        self.in_channels = in_channels
        self.device = device
        self.kernel_size = 1+6*int(sigma)
        
        self.conv = nn.Conv3d(1, in_channels, self.kernel_size, stride=1, padding=self.kernel_size//2,
                              dilation=1, bias=False, padding_mode='replicate').to(self.device)
        self.set_weights(sigma)
    
    def set_weights(self, sigma):
        sigma = torch.tensor([0.1, torch.tensor([sigma, 10]).min()]).max() # 0.1 < sigma < 10
        kernel1d = torch.arange(self.kernel_size).type(torch.float32) - self.kernel_size//2
        kernel1d = kernel1d.to(self.device)
        kernel1d = (kernel1d)**2 / 2 / sigma**2
        kernel1d = torch.exp(-kernel1d)
        kernel3d = torch.einsum('i,j,k->ijk', kernel1d, kernel1d, kernel1d)
        kernel3d /= kernel3d.sum()
        kernel3d = torch.cat(self.in_channels*[kernel3d.unsqueeze(0),], axis=0).unsqueeze(0)
        self.conv.weight = torch.nn.Parameter(kernel3d, requires_grad=False)
        
    def forward(self, vol, sigma=None):
        #print(vol.is_cuda, sigma.is_cuda, self.conv.weight.is_cuda)
        if sigma is not None:
            self.set_weights(sigma)
        return self.conv(vol)


class HessianTorch(nn.Module):
    def __init__(self, sigma, device, in_channels=1):
        super(HessianTorch, self).__init__()
        self.gauss = GaussianBlur3D(sigma=sigma, device=device, in_channels=in_channels)
        
    def forward(self, vol, sigma):
        axes = [2, 3, 4]
        gaussian_filtered = self.gauss(vol, sigma)
        
        gradient = torch.gradient(gaussian_filtered, dim=axes)
        H_elems = [torch.gradient(gradient[ax0-2], axis=ax1)[0]
              for ax0, ax1 in combinations_with_replacement(axes, 2)]
        return torch.stack(H_elems)


class HessBlock(nn.Module):
    def __init__(self, start_scale, device, act=nn.Sigmoid(), in_channels=1): #start scale - experimentaly
        super(HessBlock, self).__init__()
        
        self.device = device
        self.scale = nn.parameter.Parameter(data=torch.tensor(start_scale, dtype=torch.float32))
        
        self.linear = nn.Sequential(
            nn.Linear(7, 10, bias=True),
            nn.ReLU(),
            #nn.Linear(10, 10, bias=True),
            #nn.ReLU(),
            nn.Linear(10, 1, bias=True),
            act
        )
        self.hess = HessianTorch(self.scale, device, in_channels=in_channels)
        self.flat = nn.Flatten(start_dim=1, end_dim=4)
        
        
    def forward(self, x):
        input_sizes = x.shape
        x = self.hess(x, self.scale).permute(1,2,3,4,5,0) #1
        x = self.flat(x)
        scale_attention = self.scale*torch.ones((x.shape[0], x.shape[1], 1)).to(self.device)
        x = torch.cat([x, scale_attention], axis=2)
        x = self.linear(x) #2
        x = torch.unflatten(x, 1, input_sizes[2:])
        x = x.permute(0,4,1,2,3)
        return x
    
    
def nn_detect(vol, scale, l_func, device='cpu'):
    H = HessianTorch(scale, device)(vol, scale).permute(1,2,3,4,5,0)
    H = torch.flatten(H, start_dim=1, end_dim=4)
    scale_attention = scale*torch.ones((H.shape[0], H.shape[1], 1)).to(device)
    x = torch.cat([H, scale_attention], axis=2)
    x = l_func(x)
    x = torch.unflatten(x, 1, vol.shape[2:])
    x = x.permute(0,4,1,2,3)
    return x

    
class HessNet(nn.Module):
    def __init__(self, start_scale, device):
        super(HessNet, self).__init__()
        
        
        self.device = device
        self.H1 = HessBlock(start_scale=start_scale, device=device, act=nn.ReLU())
        self.H2 = HessBlock(start_scale=start_scale, device=device, act=nn.ReLU())
        self.H3 = HessBlock(start_scale=start_scale, device=device, act=nn.ReLU())
        self.H4 = HessBlock(start_scale=start_scale, device=device, act=nn.ReLU())
        #self.dilated_conv = MultuScaleDilationConvBlock(3, channel_coef)
        #self.conv = ConvBlock(5, 1, act = nn.Indentity())
        #self.act = nn.Sigmoid()
        
        self.out_block = nn.Sequential(
            nn.Conv3d(in_channels=5, out_channels=1, kernel_size=5, stride=1,
                      padding=2, dilation=1, padding_mode="reflect"),
            nn.Sigmoid())

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
        self.H1 = HessBlock(start_scale=start_scale, device=device, act=nn.ReLU())
        self.H2 = HessBlock(start_scale=start_scale, device=device, act=nn.ReLU())
        self.H3 = HessBlock(start_scale=start_scale, device=device, act=nn.Sigmoid())
        self.H4 = HessBlock(start_scale=start_scale, device=device, act=nn.ReLU())
        #self.dilated_conv = MultuScaleDilationConvBlock(3, channel_coef)
        #self.conv = ConvBlock(5, 1, act = nn.Indentity())
        #self.act = nn.Sigmoid()
        
        self.input_block = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1,
                      padding=1, dilation=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1,
                      padding=1, dilation=1, padding_mode="reflect"))
        
        # self.inter_block = nn.Sequential(
        #     nn.Conv3d(in_channels=9, out_channels=5, kernel_size=5, stride=1,
        #               padding=2, dilation=1, padding_mode="reflect"),
        #     nn.ReLU())
        
        self.out_block = nn.Sequential(
            nn.Conv3d(in_channels=5, out_channels=1, kernel_size=5, stride=1,
                      padding=2, dilation=1, padding_mode="reflect"),
            nn.Sigmoid())

    def forward(self, x_input):
        x = self.input_block(x_input)
        h1 = self.H1(x)
        h2 = self.H2(x)
        x = torch.cat([x, h1, h2], 1)
        x = self.inter_block(x)
        #h3 = self.H3(h1)
        #h4 = self.H4(x)
        #x = torch.cat([h1, h2, h3, h4], 1)
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