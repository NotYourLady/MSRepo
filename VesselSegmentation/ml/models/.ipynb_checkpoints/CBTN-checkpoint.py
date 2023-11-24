import torch
import torch.nn as nn

class Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, in_channels, 
                 img_shape, patch_shape,
                 hidden_size=None, drop=0.0):
        super(Embeddings, self).__init__()

        self.patch_grid_shape = torch.tensor(img_shape)//torch.tensor(patch_shape)
        if not torch.prod(torch.tensor(img_shape)==self.patch_grid_shape * torch.tensor(patch_shape)):
            raise RuntimeError(
                f"Embeddings::__init__::ERROR: Bad <img_shape>({img_shape}) or <patch_shape>({patch_shape})"
            )
        assert torch.prod(torch.tensor(img_shape)==self.patch_grid_shape * torch.tensor(patch_shape))
        self.n_patches = torch.prod(self.patch_grid_shape)

        
        if len(img_shape) == 3:
            self.is3d=True
        else:
            self.is3d=False
            
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            self.hidden_size = torch.prod(torch.tensor(patch_shape)).item()
        
        if self.is3d:
            self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                              out_channels=self.hidden_size,
                                              kernel_size=patch_shape,
                                              stride=patch_shape)
        else:
            self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                              out_channels=self.hidden_size,
                                              kernel_size=patch_shape,
                                              stride=patch_shape)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.hidden_size))
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden, patches_n)
        x = x.flatten(2) # (B, hidden, patches_n)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return x


class Projector(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU()):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=True)
        #self.fc2 = nn.Linear(out_features, out_features, bias=True)
        self.act = act
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        #x = self.act(self.fc2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(SelfAttention, self).__init__()
        
        self.query = Projector(in_features, out_features)
        self.key = Projector(in_features, out_features)
        self.value = Projector(in_features, out_features)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        logits = torch.einsum('bij,bkj->bik', q, k)
        scores = self.softmax(logits)
        out = torch.einsum('bij,bjk->bik', scores, v)
        return(out)


class MultyHeadSelfAttention(nn.Module):
    def __init__(self, in_features, n_heads):
        super(MultyHeadSelfAttention, self).__init__()

        head_features = in_features//n_heads
        self.att_list = nn.ModuleList([SelfAttention(in_features, head_features) for _ in range(n_heads)])
        
    def forward(self, x):
        out_list = []
        for att in self.att_list:
            out_list.append(att(x))
        out = torch.cat(out_list, -1)
        return(out)


class Mlp(nn.Module):
    def __init__(self, in_features, mlp_dim,
                 act=torch.nn.functional.gelu, drop=0.0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, in_features)
        self.act = act
        self.dropout = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, in_features, attention_heads, mlp_dim, drop=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_features, eps=1e-6)
        self.norm2 = nn.LayerNorm(in_features, eps=1e-6)
        self.mlp = Mlp(in_features, mlp_dim, drop=drop)
        self.attn = MultyHeadSelfAttention(in_features, attention_heads)

    def forward(self, x):
        x = x + self.attn(x)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm1(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 in_features,
                 attention_heads,
                 mlp_dim,
                 transformer_layers,
                 drop=0.0):
        super(TransformerEncoder, self).__init__()
        self.block_list = nn.ModuleList()
        self.norm = nn.LayerNorm(in_features, eps=1e-6)
        for _ in range(transformer_layers):
            self.block_list.append(TransformerBlock(in_features=in_features,
                                   attention_heads=attention_heads,
                                   mlp_dim=mlp_dim, drop=drop))

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        out = self.norm(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 in_channels,
                 img_shape,
                 patch_shape,
                 hidden_size,
                 attention_heads,
                 mlp_dim, 
                 transformer_layers,
                 drop=0.0):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(in_channels=in_channels, img_shape=img_shape,
                                     patch_shape=patch_shape, hidden_size=hidden_size, drop=drop)
        self.encoder = TransformerEncoder(in_features=self.embeddings.hidden_size, 
                                          attention_heads=attention_heads,
                                          mlp_dim=mlp_dim,
                                          transformer_layers=transformer_layers,
                                          drop=drop)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  
        return encoded


class Conv3dReLU(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels,
                         kernel_size, stride=stride,
                         padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm3d(out_channels)
        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 skip_channels=0,
                 use_batchnorm=True,
                 resample=1,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.resample = nn.Upsample(scale_factor=resample, mode='trilinear', align_corners=None)


    def forward(self, x, skip=None):
        x = self.resample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self,
                 hidden_size,
                 out_channels,
                 patch_grid_shape,
                 skip_channels=0,
                 resample=2):
        super().__init__()
        
        self.patch_grid_shape = patch_grid_shape
        
        blocks = [
            DecoderBlock(in_channels=hidden_size, out_channels=hidden_size//2, resample=2),
            DecoderBlock(in_channels=hidden_size//2, out_channels=hidden_size//4, resample=2),
            DecoderBlock(in_channels=hidden_size//4, out_channels=hidden_size//8, resample=2),
            DecoderBlock(in_channels=hidden_size//8, out_channels=out_channels, resample=2),
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        H, W, D = self.patch_grid_shape
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, H, W, D)
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x, skip=None)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_shape=(64, 64, 64),
                 patch_shape=(16, 16, 16),
                 in_channels=1,
                 hidden_size=512,
                 transformer_layers=8,
                 attention_heads=8,
                 mlp_dim=1024, 
                 drop=0.0):
        super(VisionTransformer, self).__init__()
        self.transformer = Transformer(in_channels=in_channels,
                                       img_shape=img_shape,
                                       patch_shape=patch_shape,
                                       hidden_size=hidden_size,
                                       attention_heads=attention_heads,
                                       mlp_dim=mlp_dim, 
                                       transformer_layers=transformer_layers)
        self.decoder = DecoderCup(hidden_size=self.transformer.embeddings.hidden_size,
                                  out_channels=1,
                                  patch_grid_shape=self.transformer.embeddings.patch_grid_shape)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x)
        return self.sigmoid(x)

