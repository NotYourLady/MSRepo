import torch
import torch.nn as nn


def check_None(tensor):
    if torch.isnan(tensor).sum() > 0:
        raise RuntimeError(f"None here ({torch.isnan(tensor).sum()})")


class swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)


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
        check_None(normed_x)
        out = self.act(normed_x)
        return out


class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super(conv_block, self).__init__()
        self.act = norm_act(in_channels)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              padding_mode="reflect")

    def forward(self, x):
        check_None(x)
        x = self.act(x)
        check_None(x)
        x = self.conv(x)
        check_None(x)
        return x

    
class stem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super(stem, self).__init__()
        
        self.main_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  padding_mode="reflect"),
            conv_block(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding)
        )
        
        self.shortcut_conv =  nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(1,1,1), stride=1, padding=0),
            norm_act(out_channels, nn.Identity())
        )
        
        
    def forward(self, x):
        main = self.main_conv(x)
        shortcut = self.shortcut_conv(x)
        return main + shortcut

        

class residual_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 stride=1,
                 drop=0):
        super(residual_block, self).__init__()
        
        self.main_conv = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            conv_block(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=1, padding=1)
        )
        self.shortcut_conv =  nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(1,1,1), stride=stride, padding=0,
                                  padding_mode="replicate"),
            norm_act(out_channels, nn.Identity())
        )
        self.dropout = nn.Dropout3d(p=drop)

    def forward(self, x):
        main = self.main_conv(x)
        shortcut = self.shortcut_conv(x)
        out = main + shortcut
        return self.dropout(out)


class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 stride=2, padding=2,
                 drop=0):
        super(upsample_block, self).__init__()
        
        self.unconv = torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                               padding=1, output_padding=stride-1)
        
    def forward(self, x):
        out = self.unconv(x)
        return out
    
    
class attention_gate(nn.Module):
    def __init__(self, in_channels1, in_channels2, intermediate_channels, act=nn.PReLU()):
        super(attention_gate, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=in_channels1, out_channels=intermediate_channels,
                              kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=in_channels2, out_channels=intermediate_channels,
                              kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv3d(in_channels=intermediate_channels, out_channels=1,
                              kernel_size=1, stride=1, padding=0)
        self.act = act
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        x1_conv = self.conv1(x1)
        x2_conv = self.conv2(x2)
        inter = self.act(x1_conv + x2_conv)
        inter = self.sigmoid(self.conv(inter))
        return x1*inter


class attention_concat(nn.Module):
    def __init__(self, main_channels, skip_channels, with_attention=True):
        super(attention_concat, self).__init__()
        self.with_attention = with_attention
        if with_attention:
            self.att_gate = attention_gate(skip_channels, main_channels, main_channels)
    
    def forward(self, main, skip):
        if self.with_attention:
            attention_across = self.att_gate(skip, main)
            return torch.cat([main, attention_across], dim=1)
        else:
            return torch.cat([main, skip], dim=1)


class downsample(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1,
                 bias=False, drop=0.0,
                 use_spec_norm=False, act=nn.ReLU()):
        super(downsample, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              padding_mode="reflect")
        self.act = act
        
        if use_spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
            self.norm = nn.Identity()
        else:
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.dropout = nn.Dropout3d(p=drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)


class VG_discriminator(nn.Module):
    def __init__(self, in_channels=1, channels_coef=64,
                 use_spec_norm=False, act=nn.ReLU()):
        super(VG_discriminator, self).__init__()
            
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=channels_coef,
                              kernel_size=4, stride=2, padding=1,
                              padding_mode="reflect")
        
        if use_spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
            self.norm = nn.Identity()
        else:
            self.norm = nn.InstanceNorm3d(channels_coef, affine=True)
        self.act = act
        
        self.downsample = nn.Sequential(
            downsample(channels_coef, 2*channels_coef, kernel_size=4, stride=1, padding='same', act=nn.PReLU()),
            downsample(2*channels_coef, 4*channels_coef, kernel_size=4, stride=2, padding=1, act=nn.PReLU()),
            downsample(4*channels_coef, 8*channels_coef, kernel_size=4, stride=2, padding=1, act=nn.PReLU())
        )
        self.conv2 = nn.Conv3d(in_channels=8*channels_coef, out_channels=1,
                               kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        
        x = self.downsample(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x
