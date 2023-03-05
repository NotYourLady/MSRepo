#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from functools import partial
import torch
import torch.nn as nn


def get_noise(size, device, noise_coef=1.0e-2):
    noise =  noise_coef * torch.randn(*size, device=device)
    return(noise)    


def get_noised_labels(size, type_, device, noise_coef=1.0e-2):
    assert type_!=1 or type_!=0
    noise = torch.abs(get_noise(size, device, noise_coef))
    if type_==1:
        labels = torch.ones(size, device=device)
        labels -= noise
        return(labels)
    if type_==0:
        labels = torch.zeros(size, device=device)
        labels += noise
        return(labels)


class Noiser(nn.Module):
    def __init__(self, device, noise_coef=1.0e-3):
        super().__init__()
        self.noise_coef_ = noise_coef
        self.device_ = device
        
    def forward(self, x):
        x = x + torch.randn(*(x.shape), device= self.device_)
        return x

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)   


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

        
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class Discriminator128(nn.Module):
    def __init__(self, drop=0.0):
        super().__init__()
        
        self.core = nn.Sequential(
            # in: 3 x 128 x 128

            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 32 x 64 x 64
            ResNetBasicBlock(32, 32), 
            # out: 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 4  x 4

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 1 x 1


            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid() )
        
    def forward(self, x):
        assert x.shape[-1] == 128
        x = self.core(x)
        return x


class Generator128(nn.Module):
    def __init__(self, latent_size, noiser_coef=0.003, drop=0.0):
        super().__init__()
        
        self.noiser = Noiser(noiser_coef)

        self.core = nn.Sequential(
            # in: latent_size x 1 x 1

            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            ResNetBasicBlock(512, 1024),
            #nn.Dropout2d(drop),
            # 1024 x 4 x 4

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 8 x 8
            ResNetBasicBlock(512, 256),
            #nn.Dropout2d(drop_gen)
            # 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16
            ResNetBasicBlock(128, 128),
            #nn.Dropout2d(drop_gen),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32
            self.noiser,
            ResNetBasicBlock(64, 64),
            #nn.Dropout2d(drop_gen),
            # out: 64 x 32 x 32
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # out: 32 x 64 x 64
            ResNetBasicBlock(32, 32),
            # out: 32 x 64 x 64
            
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 1 x 128 x 128
        )
    def forward(self, x):
        x = self.core(x)
        return x

