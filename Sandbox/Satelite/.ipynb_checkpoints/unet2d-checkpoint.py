import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, act_fn=nn.ReLU(inplace=True)):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            act_fn,
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
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
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            act_fn)

    def forward(self, x):
        x = self.up(x)
        return x



class Unet2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=16,
                 act=nn.Sigmoid(), depth=4):
        super(Unet2d, self).__init__()
        
        self.depth = depth
        
        filters = [in_channels,]
        for i in range(depth+1):
            filters.append(channels * (2**i))
        
        self.Pools = nn.ModuleList([])
        self.Convs = nn.ModuleList([])
        self.UpConvs = nn.ModuleList([])
        self.Ups = nn.ModuleList([])

        for i in range(1, depth+1):
            self.Pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.Convs.append(conv_block(filters[i], filters[i+1]))
            self.UpConvs.append(conv_block(filters[i+1], filters[i]))
            self.Ups.append(up_conv(filters[i+1], filters[i]))

        self.inConv = conv_block(in_channels, channels)
        self.outConv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.act = act
        
    def forward(self, x):
        #x -= x.mean()
        #x /=x.std()
        down_features = []    
        down_features.append(self.inConv(x))
        
        for i in range(self.depth):
            down_features.append(self.Convs[i](self.Pools[i](down_features[i])))


        x = down_features[i+1]
        for i in reversed(range(self.depth)):
            x = self.Ups[i](x)
            x = torch.cat((down_features[i], x), dim=1)
            x = self.UpConvs[i](x)

        out = self.outConv(x) 
        out = self.act(out)
        return out
