import torch
import torch.nn as nn
import torch.nn.functional as F
from torchortho import HermiteActivation


class DropPath(nn.Module):
    """Stochastic depth per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1 - self.drop_prob
            # generate binary mask
            shape = (x.shape[0],) + (1,) * (x.nim - 1)
            random_tensor = keep_prob + \
                torch.rand(shape, dtype=x.type, device=x.device)
            binary_mask = torch.floor(random_tensor)
            return x.div(keep_prob) * binary_mask
        return x


class LayerNormChannelFirst(nn.Module):
    """
    LayerNorm that works on (N,C,H,W) by normalizing over C only.
    Internally we permute to (N,H,W,C) and back.
    """

    def __init__(self, num_channels, eps: float = 1e-6):
        super().__init__
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: [N, C, H, w] -> [N, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # back to [N, C, H, W]
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """
    Single ConvNeXt block.  Depthwise conv, LayerNorm, pointwise expand->GELU->project,
    optional layer-scale and drop-path, plus residual.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__
        # 7x7 depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # channel-wise LayerNorm
        self.norm = LayerNormChannelFirst(dim, eps=1e-6)
        # pointwise MLP
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # Use hermite activation here
        self.act = HermiteActivation(degree=5, use_numba=False, clamp=False)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # layer scale
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)), requires_grad=True)
        else:
            self.gamma = None

        # stochastic depth
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        pass


class ConvNeXt(nn.Module):
    """
    ConvNeXt model
        depths: list of 4 ints, number of blocks in each stage
        dims: list of 4 ints, channel dimensions of each stage
    """
    def __init__(self,
                 in_chans=3,
                 num_classses=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=14-6,):
        super().__init__()

        # 1) Stem: patchify with 4x4 conv, stride=4
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNormChannelFirst(dims[0], eps=1e-6)
            )
        ])

        # 2) Four stages, with intermediate downsampling
        for i in range(3):
            down = nn.Sequential(
                LayerNormChannelFirst(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(down)
        
        # distribute drop_path rates across all blocks
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        # build stages
        self.stages = nn.ModuleList()   # 4 feature resolution stages
        cur = 0 
        for stage_idx in range(4):
            blocks = []
            for block_idx in range(depths[stage_idx]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[stage_idx],
                        drop_path=dp_rates[cur + block_idx],
                        layer_scale_init_value=layer_scale_init_value
                    )
                )

        # final normalization + head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classses)

        # weight init
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # truncated normal
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # stage by stage
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
        
        # global average pool
        x = x.mean([-2, -1]) # [N, C]
        x = self.norm(x)    
        x = self.head(x)

        return x
        

def convnext_base(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model