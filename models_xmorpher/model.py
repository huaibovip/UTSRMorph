# Copyright (c) MMIPT. All rights reserved.
import torch
from einops import rearrange
from mmcv.cnn import build_conv_layer
from mmengine.model import normal_init
from torch import nn
from torch.nn import functional as F

from .module import BasicLayer, PatchEmbed3D, PatchExpand, PatchMerging
from .module_swin import BasicLayer as BasicLayerSwin


class XMorpher(nn.Module):
    """
    Structure: 4 encoding stages(BasicLayer) + 4 decoding stages(BasicLayerUp)

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_dim (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(
        self,
        img_size,
        window_size=[2, 2, 2],
        in_dim=1,
        embed_dim=48,
        patch_size=[4, 4, 4],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        basic_layer=BasicLayer,
        patch_norm=False,
        frozen_stages=-1,
        use_checkpoint=False,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            in_dim=in_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = basic_layer(
                embed_dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers -
                1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.up_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in reversed(range(self.num_layers)):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2**i_layer),
                int(embed_dim * 2**i_layer),
            )

            up_layer = basic_layer(
                embed_dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchExpand if i_layer > 0 else None,
                use_checkpoint=use_checkpoint,
            )
            self.up_layers.append(up_layer)
            self.concat_back_dim.append(concat_linear)

        self.num_features = int(embed_dim * 2**(self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.norm2 = norm_layer(self.embed_dim * 2)

        self.reverse_patch_embedding = nn.ConvTranspose3d(
            2 * embed_dim,
            embed_dim // 2,
            kernel_size=(4, 4, 4),
            stride=4,
        )
        self.flow = DefaultFlow(24)
        self.warp = Warp(img_size)

    def forward(self, moving, fixed, **kwargs):
        moving = self.patch_embed(moving)
        fixed = self.patch_embed(fixed)

        moving = self.pos_drop(moving)
        fixed = self.pos_drop(fixed)

        moving = rearrange(moving, 'n c d h w -> n d h w c')
        fixed = rearrange(fixed, 'n c d h w -> n d h w c')

        features_moving = []
        features_fixed = []
        for layer in self.layers:
            moving_out, fixed_out, moving, fixed = layer(
                moving.contiguous(), fixed.contiguous())
            features_moving.append(moving_out)
            features_fixed.append(fixed_out)

        moving = self.norm(moving)
        fixed = self.norm(fixed)

        for inx, layer_up in enumerate(self.up_layers):
            if inx == 0:
                _, _, moving, fixed = layer_up(moving, fixed)
            else:
                if moving.shape != features_moving[3 - inx].shape:
                    moving = rearrange(moving, 'n d h w c -> n c d h w')
                    fixed = rearrange(fixed, 'n d h w c -> n c d h w')
                    B, D, W, H, C = features_moving[3 - inx].shape
                    moving = F.interpolate(moving,
                                           size=(D, W, H),
                                           mode='trilinear',
                                           align_corners=True)
                    fixed = F.interpolate(fixed,
                                          size=(D, W, H),
                                          mode='trilinear',
                                          align_corners=True)
                    moving = rearrange(moving, 'n c d h w -> n d h w c')
                    fixed = rearrange(fixed, 'n c d h w -> n d h w c')

                moving = torch.cat([moving, features_moving[3 - inx]], -1)
                fixed = torch.cat([fixed, features_fixed[3 - inx]], -1)
                moving = self.concat_back_dim[inx](moving)
                fixed = self.concat_back_dim[inx](fixed)
                _, _, moving, fixed = layer_up(moving, fixed)

        x = torch.cat([moving, fixed], -1)
        x = self.norm2(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        # reverse the patch embedding to transfer the final feature into image size
        x = self.reverse_patch_embedding(x)
        flow = self.flow(x)

        warped = self.warp(moving, flow=flow)
        return warped, flow


class XMorpherSwin(XMorpher):

    def __init__(self,
                 window_size,
                 in_dim=1,
                 embed_dim=48,
                 patch_size=(4, 4, 4),
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0,
                 attn_drop_rate=0,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 basic_layer=BasicLayerSwin,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__(window_size, in_dim, embed_dim, patch_size, depths,
                         num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                         attn_drop_rate, drop_path_rate, norm_layer,
                         basic_layer, patch_norm, frozen_stages,
                         use_checkpoint)


class Warp(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i,
                     ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DefaultFlow(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels=3,
        kernel_size=3,
        bias=True,
    ) -> None:
        super().__init__()

        self.conv = build_conv_layer(
            cfg=dict(type=f'Conv{out_channels}d'),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=1,
            bias=bias,
        )
        self.init_weights()

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.conv(x)
        return x

    def init_weights(self):
        normal_init(self.conv, mean=0, std=1e-5, bias=0)
