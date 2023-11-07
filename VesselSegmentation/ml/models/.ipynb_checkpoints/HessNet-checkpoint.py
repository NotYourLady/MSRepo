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
        if type(start_scale) != torch.Tensor:
            scale = torch.tensor(start_scale*1, dtype=torch.float32, device=device)
        else:
            scale = start_scale.clone().to(device)
            
        if scale.shape == torch.Size([]):
            scale = scale.unsqueeze(0)
        self.scale = nn.parameter.Parameter(data=scale)
        
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
        
        
class HessFeatures(nn.Module):
    def __init__(self, start_scale, device, out_act=nn.Sigmoid(), in_channels=1, n_hess_blocks=4):
        super(HessFeatures, self).__init__()    
        self.device = device
        
        self.HessBlocks = nn.ModuleList([])
        for i in range(n_hess_blocks):
            self.HessBlocks.append(HessBlock(start_scale=(0.5+i/2)*torch.tensor(start_scale),
                                            device=device, act=nn.ReLU(),
                                            in_channels=in_channels))
        
    def forward(self, x):
        h = []
        for HessBlock in self.HessBlocks:
            h.append(HessBlock(x))     
        out = torch.cat(h, 1)
        return(out)
    
    def to_device(self, device):
        self.to(device)
        for HessBlock in self.HessBlocks:
            HessBlock.device = device
        
        for HessBlock in self.HessBlocks:
            HessBlock.hess.gauss.device = device       

            
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, act_fn=nn.ReLU(inplace=True)):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn,
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, in_ch, out_ch):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1,
                 padding=1, bias=True, act_fn=nn.ReLU(inplace=True)):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn)

    def forward(self, x):
        x = self.up(x)
        return x            
            

class HessUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=16,
                 act=nn.Sigmoid(), depth=3, device='cuda'):
        super(HessUNet, self).__init__()
        
        self.depth = depth
        
        filters = [in_channels,]
        for i in range(depth+1):
            filters.append(channels * (2**i))
        
        self.Pools = nn.ModuleList([])
        self.Convs = nn.ModuleList([])
        self.UpConvs = nn.ModuleList([])
        self.Ups = nn.ModuleList([])

        for i in range(1, depth+1):
            self.Pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.Convs.append(conv_block(filters[i], filters[i+1]))
            self.UpConvs.append(conv_block(filters[i+1], filters[i]))
            self.Ups.append(up_conv(filters[i+1], filters[i]))

        
        N_hess_blocks=4
        self.HessFeatures = HessFeatures(start_scale=torch.tensor([0.8, 0.8, 1.2]), device=device,
                                         out_act=nn.ReLU(), n_hess_blocks=N_hess_blocks,
                                         in_channels=in_channels)
        self.inConv = conv_block(filters[0], filters[1]-N_hess_blocks)
        
        #self.inConv = conv_block(filters[0], filters[1])
        self.outConv = nn.Conv3d(channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = act
        
    def forward(self, x):
        down_features = []    
        
        down_features.append(torch.cat((self.HessFeatures(x), self.inConv(x)), dim=1))
        #down_features.append(self.inConv(x))
        
        for i in range(self.depth):
            down_features.append(self.Convs[i](self.Pools[i](down_features[i])))

        for i in reversed(range(self.depth)):
            x = self.Ups[i](down_features[i+1])
            x = torch.cat((down_features[i], x), dim=1)
            x = self.UpConvs[i](x)

        out = self.outConv(x)
        out = self.act(out)
        return out
    

class HessUNet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=16,
                 act=nn.Sigmoid(), depth=3, device='cuda'):
        super(HessUNet2, self).__init__()
        
        self.depth = depth
        
        filters = [in_channels,]
        for i in range(depth+1):
            filters.append(channels * (2**i))
        
        self.Pools = nn.ModuleList([])
        self.Convs = nn.ModuleList([])
        self.UpConvs = nn.ModuleList([])
        self.Ups = nn.ModuleList([])

        for i in range(1, depth+1):
            self.Pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.Convs.append(conv_block(filters[i], filters[i+1]))
            self.UpConvs.append(conv_block(filters[i+1], filters[i]))
            self.Ups.append(up_conv(filters[i+1], filters[i]))

        
        N_hess_blocks=4
        self.HessFeatures_in = HessFeatures(start_scale=torch.tensor([0.8, 0.8, 1.2]), device=device,
                                         out_act=nn.ReLU(), n_hess_blocks=N_hess_blocks,
                                         in_channels=in_channels)
        self.HessFeatures_out = HessFeatures(start_scale=torch.tensor([0.8, 0.8, 1.2]), device=device,
                                         out_act=nn.ReLU(), n_hess_blocks=N_hess_blocks,
                                         in_channels=out_channels)
        
        self.inConv = conv_block(filters[0], filters[1]-N_hess_blocks)
        self.outConv1 = nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.outConv2 = nn.Conv3d(out_channels+N_hess_blocks, out_channels,
                                  kernel_size=5, stride=1, padding=2)
        self.act = act
        
    def forward(self, x):
        down_features = []    
        down_features.append(torch.cat([self.HessFeatures_in(x), self.inConv(x)], dim=1))
        
        for i in range(self.depth):
            down_features.append(self.Convs[i](self.Pools[i](down_features[i])))

        for i in reversed(range(self.depth)):
            x = self.Ups[i](down_features[i+1])
            x = torch.cat((down_features[i], x), dim=1)
            x = self.UpConvs[i](x)

        out = self.outConv1(x)     
        out = torch.cat([self.HessFeatures_out(out), out], dim=1)
        out = self.outConv2(out)
        out = self.act(out)
        return out
    