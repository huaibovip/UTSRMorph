# Copyright (c) MMIPT. All rights reserved.
from functools import partial, reduce
from operator import mul

import torch
import torch.utils.checkpoint as checkpoint
from mmcv.cnn.bricks.drop import DropPath
from mmengine.model.weight_init import trunc_normal_
from mmengine.utils import to_ntuple
from torch import Tensor, nn

import torch
from mmcv.cnn import build_conv_layer
from mmengine.model import normal_init
from torch import Tensor, nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        ndim,
        img_size,
        in_chans,
        hidden_dim=16,
        patch_size=4,
        embed_dim=96,
        patch_way=None,
    ):
        super().__init__()
        img_size = to_ntuple(ndim)(img_size)
        patch_size = to_ntuple(ndim)(patch_size)
        patches_res = [img_size[i] // patch_size[i] for i in range(ndim)]
        self.num_patches = reduce(mul, patches_res)
        self.img_size = img_size
        self.ndim = ndim

        assert patch_way in [
            "overlaping",
            "nonoverlaping",
            "pointconv",
        ], "the patch embedding way isn't exist!"

        if patch_way == "nonoverlaping":
            self.proj = getattr(nn, f'Conv{ndim}d')(in_chans,
                                                    embed_dim,
                                                    kernel_size=patch_size,
                                                    stride=patch_size)
        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                getattr(nn, f'Conv{ndim}d')(in_chans,
                                            hidden_dim,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),  # D, H, W
                getattr(nn, f'BatchNorm{ndim}d')(hidden_dim),
                nn.GELU(),
                getattr(nn, f'Conv{ndim}d')(
                    hidden_dim,
                    int(hidden_dim * 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),  # D//2, H//2, W//2
                getattr(nn, f'BatchNorm{ndim}d')(int(hidden_dim * 2)),
                nn.GELU(),
                getattr(nn, f'Conv{ndim}d')(
                    int(hidden_dim * 2),
                    int(hidden_dim * 4),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),  # D//2, H//2, W//2
                getattr(nn, f'BatchNorm{ndim}d')(int(hidden_dim * 4)),
                nn.GELU(),
                getattr(nn, f'Conv{ndim}d')(
                    int(hidden_dim * 4),
                    embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),  # D//4, H//4, W//4
            )
        else:
            self.proj = nn.Sequential(
                getattr(nn, f'Conv{ndim}d')(in_chans,
                                            hidden_dim,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False),  # D//2, H//2, W//2
                getattr(nn, f'BatchNorm{ndim}d')(hidden_dim),
                nn.GELU(),
                getattr(nn, f'Conv{ndim}d')(
                    hidden_dim,
                    int(hidden_dim * 2),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),  # D//2, H//2, W//2
                getattr(nn, f'BatchNorm{ndim}d')(int(hidden_dim * 2)),
                nn.GELU(),
                getattr(nn, f'Conv{ndim}d')(
                    int(hidden_dim * 2),
                    int(hidden_dim * 4),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),  # D//4, H//4, W//4
                getattr(nn, f'BatchNorm{ndim}d')(int(hidden_dim * 4)),
                nn.GELU(),
                getattr(nn, f'Conv{ndim}d')(
                    int(hidden_dim * 4),
                    embed_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),  # D//4, H//4, W//4
            )

    def forward(self, x: Tensor):
        # B, C, D, H, W = x.shape
        assert all([
            x.shape[2:][i] == self.img_size[i] for i in range(self.ndim)
        ]), (f"Input image size ({reduce(mul, x.shape[2:])}) "
             f"doesn't match model ({reduce(mul, self.img_size)}).")

        x = self.proj(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        merging_way,
        cpe_per_satge,
    ):
        super().__init__()
        assert merging_way in [
            "conv3_2",
            "conv2_2",
            "avgpool3_2",
            "avgpool2_2",
        ], "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge
        if merging_way == "conv3_2":
            self.proj = nn.Sequential(
                getattr(nn, f'Conv{ndim}d')(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1),
                getattr(nn, f'BatchNorm{ndim}d')(out_channels),
            )
        elif merging_way == "conv2_2":
            self.proj = nn.Sequential(
                getattr(nn, f'Conv{ndim}d')(in_channels,
                                            out_channels,
                                            kernel_size=2,
                                            stride=2,
                                            padding=0),
                getattr(nn, f'BatchNorm{ndim}d')(out_channels),
            )
        elif merging_way == "avgpool3_2":
            self.proj = nn.Sequential(
                getattr(nn, f'AvgPool{ndim}d')(in_channels,
                                               out_channels,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1),
                getattr(nn, f'BatchNorm{ndim}d')(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                getattr(nn, f'AvgPool{ndim}d')(in_channels,
                                               out_channels,
                                               kernel_size=2,
                                               stride=2,
                                               padding=0),
                getattr(nn, f'BatchNorm{ndim}d')(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = getattr(nn, f'Conv{ndim}d')(out_channels,
                                                         out_channels,
                                                         3,
                                                         padding=1,
                                                         groups=out_channels)

    def forward(self, x: Tensor):
        # B, C, D, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.m_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.m_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.f_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.f_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.m_proj = nn.Linear(dim, dim)
        self.f_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, mov: Tensor, fix: Tensor):
        # B, D, H, W, C = mov.shape
        shapes = mov.shape
        B, img_sizes, C = shapes[0], shapes[1:-1], shapes[-1]
        mul_size = reduce(mul, img_sizes)  # [D] * H * W

        shape1 = (B, mul_size, 2, self.num_heads, C // self.num_heads)
        shape2 = (B, mul_size, 1, self.num_heads, C // self.num_heads)
        mov_kv = self.m_kv(mov).reshape(shape1).permute(2, 0, 3, 1, 4)
        fix_kv = self.f_kv(fix).reshape(shape1).permute(2, 0, 3, 1, 4)
        mov_q = self.m_q(mov).reshape(shape2).permute(2, 0, 3, 1, 4)
        fix_q = self.f_q(fix).reshape(shape2).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        mov_Q, mov_K, mov_V = mov_q[0], mov_kv[0], mov_kv[1]
        fix_Q, fix_K, fix_V = fix_q[0], fix_kv[0], fix_kv[1]

        mov_attn = (fix_Q @ mov_K.transpose(-2, -1)) * self.scale
        fix_attn = (mov_Q @ fix_K.transpose(-2, -1)) * self.scale
        mov_attn = mov_attn.softmax(dim=-1)
        fix_attn = fix_attn.softmax(dim=-1)

        mov_attn = self.attn_drop(mov_attn)
        fix_attn = self.attn_drop(fix_attn)

        mov = (mov_attn @ mov_V).transpose(1, 2).reshape(B, *img_sizes, C)
        mov = self.m_proj(mov)
        mov = self.proj_drop(mov)

        fix = (fix_attn @ fix_V).transpose(1, 2).reshape(B, *img_sizes, C)
        fix = self.f_proj(fix)
        fix = self.proj_drop(fix)

        return mov, fix


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(
        self,
        ndim,
        dim,
        num_heads,
        qk_scale=None,
        attn_drop=0,
        kernel_size=3,
        dilation=1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = to_ntuple(ndim)(kernel_size)
        self.dilation = dilation

        kwargs = dict(kernel_size=kernel_size,
                      dilation=dilation,
                      padding=dilation * (kernel_size - 1) // 2,
                      stride=1)
        if ndim == 2:
            self.unfold = partial(nn.functional.unfold, **kwargs)
        elif ndim == 3:
            from .unfoldNd.unfold import unfoldNd
            self.unfold = partial(unfoldNd, **kwargs)
        else:
            raise ValueError(f'ndim only supports 2 and 3, but got {ndim}')
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

    def forward(self, q, k, v):
        # B, d=(C//num_dilation), [D], H, W
        shapes = q.shape
        (B, d), img_sizes = shapes[:2], shapes[2:]
        mul_size = reduce(mul, img_sizes)

        shape1 = (B, d // self.head_dim, self.head_dim, 1, mul_size)
        shape2 = (B, d // self.head_dim, self.head_dim,
                  reduce(mul, self.kernel_size), mul_size)

        # Q: (B,h,N,1,d)
        q = q.reshape(shape1).permute(0, 1, 4, 3, 2)
        # K: (B,h,N,d,k*k)
        k = self.unfold(k).reshape(shape2).permute(0, 1, 4, 2, 3)
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # V: (B,h,N,k*k,d)
        v = self.unfold(v).reshape(shape2).permute(0, 1, 4, 3, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, *img_sizes, d)
        return x


class MultiScaleDilateAttention(nn.Module):
    "Implementation of Multi-Scale Dilated Attention"

    def __init__(
        self,
        ndim,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        kernel_size=3,
        dilation=[1, 2, 3],
    ):
        super().__init__()
        self.ndim = ndim
        self.dim = dim
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num_dilation = len(dilation)

        assert (
            num_heads % self.num_dilation == 0
        ), f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        head_dim = dim // self.num_dilation
        head_num_heads = num_heads // self.num_dilation
        # qk_scale = qk_scale or head_dim**-0.5

        # self.qkv = getattr(nn, f'Conv{ndim}d')(dim, dim * 3, 1, bias=qkv_bias)
        self.m_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.f_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dilate_attn = nn.ModuleList([
            DilateAttention(
                ndim,
                head_dim,
                head_num_heads,
                qk_scale,
                attn_drop,
                kernel_size,
                dilation[i],
            ) for i in range(self.num_dilation)
        ])
        self.m_proj = nn.Linear(dim, dim)
        self.f_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, mov: Tensor, fix: Tensor):
        shapes = mov.shape  # B, D, H, W, C
        B, img_sizes, C = shapes[0], shapes[1:-1], shapes[-1]

        # (2, 1, 0, 3, 4, 5, [6])
        shaper = (B, *img_sizes, 3, self.num_dilation, C // self.num_dilation)
        # num_dilation, 2, B, C//num_dilation, D, H, W
        shapep = (2 + self.ndim, 1 + self.ndim, 0, 3 + self.ndim,
                  *list(range(1, self.ndim + 1)))
        mov_qkv = self.m_qkv(mov).reshape(shaper).permute(shapep)
        fix_qkv = self.f_qkv(fix).reshape(shaper).permute(shapep)

        m_res = []
        f_res = []
        for i in range(self.num_dilation):
            mov_i = self.dilate_attn[i](fix_qkv[i][0], mov_qkv[i][1],
                                        mov_qkv[i][2])
            fix_i = self.dilate_attn[i](mov_qkv[i][0], fix_qkv[i][1],
                                        fix_qkv[i][2])
            m_res.append(mov_i)
            f_res.append(fix_i)

        # B, *img_size, num_dilation, C//num_dilation
        mov = torch.stack(m_res, dim=self.ndim + 1).reshape(shapes)
        fix = torch.stack(f_res, dim=self.ndim + 1).reshape(shapes)
        mov = self.m_proj(mov)
        fix = self.f_proj(fix)
        mov = self.proj_drop(mov)
        fix = self.proj_drop(fix)
        return mov, fix


class MultiScaleConvDilateAttention(nn.Module):
    "Implementation of Multi-Scale Dilated Attention"

    def __init__(
        self,
        ndim,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        kernel_size=3,
        dilation=[1, 2, 3],
    ):
        super().__init__()
        self.ndim = ndim
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim**-0.5
        self.num_dilation = len(dilation)
        assert (
            num_heads % self.num_dilation == 0
        ), f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.m_qkv = getattr(nn, f'Conv{ndim}d')(dim,
                                                 dim * 3,
                                                 1,
                                                 bias=qkv_bias)
        self.f_qkv = getattr(nn, f'Conv{ndim}d')(dim,
                                                 dim * 3,
                                                 1,
                                                 bias=qkv_bias)
        self.dilate_attn = nn.ModuleList([
            DilateAttention(
                ndim,
                head_dim,
                qk_scale,
                attn_drop,
                kernel_size,
                dilation[i],
            ) for i in range(self.num_dilation)
        ])
        self.m_proj = nn.Linear(dim, dim)
        self.f_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, mov: Tensor, fix: Tensor):
        shapes = mov.shape  # B, D, H, W, C
        B, img_sizes, C = shapes[0], shapes[1:-1], shapes[-1]

        shape1 = (0, self.ndim + 1, *list(range(1, self.ndim + 1)))
        mov, fix = mov.permute(shape1), fix.permute(shape1)

        shaper = (B, 3, self.num_dilation, C // self.num_dilation, *img_sizes)
        shapep = (2, 1, 0, 3, *list(range(4, self.ndim + 4)))
        # num_dilation, 3, B, C//num_dilation, D, H, W
        mov_qkv = self.m_qkv(mov).reshape(shaper).permute(shapep)
        fix_qkv = self.f_qkv(fix).reshape(shaper).permute(shapep)

        m_res = []
        f_res = []
        for i in range(self.num_dilation):
            mov_i = self.dilate_attn[i](fix_qkv[i][0], mov_qkv[i][1],
                                        mov_qkv[i][2])
            fix_i = self.dilate_attn[i](mov_qkv[i][0], fix_qkv[i][1],
                                        fix_qkv[i][2])
            m_res.append(mov_i)
            f_res.append(fix_i)

        # B, *img_size, num_dilation, C//num_dilation
        mov = torch.stack(m_res, dim=self.ndim + 1).reshape(shapes)
        fix = torch.stack(f_res, dim=self.ndim + 1).reshape(shapes)
        mov = self.m_proj(mov)
        fix = self.f_proj(fix)
        mov = self.proj_drop(mov)
        fix = self.proj_drop(fix)
        return mov, fix


class DilateBlock(nn.Module):
    r"""Dilated Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        ndim,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        kernel_size=3,
        dilation=[1, 2, 3],
        cpe_per_block=False,
        dilate_attention=True,
    ):
        super().__init__()
        self.ndim = ndim
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block

        if self.cpe_per_block:
            self.pos_embed = getattr(nn, f'Conv{ndim}d')(dim,
                                                         dim,
                                                         3,
                                                         padding=1,
                                                         groups=dim)

        self.m_norm1 = norm_layer(dim)
        self.f_norm1 = norm_layer(dim)
        if dilate_attention:
            self.attn = MultiScaleDilateAttention(
                ndim=ndim,
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        else:
            self.attn = GlobalAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        mov, fix = x

        if self.cpe_per_block:
            fix = fix + self.pos_embed(fix)
            mov = mov + self.pos_embed(mov)

        # (0, 2, 3, 4, 1) channel_last
        shape1 = (0, *list(range(2, self.ndim + 2)), 1)
        mov = mov.permute(shape1)
        fix = fix.permute(shape1)

        # MSDA
        attn_mov, attn_fix = self.attn(self.m_norm1(mov), self.f_norm1(fix))
        mov = mov + self.drop_path(attn_mov)
        fix = fix + self.drop_path(attn_fix)

        # FFN
        mov = mov + self.drop_path(self.mlp(self.norm2(mov)))
        fix = fix + self.drop_path(self.mlp(self.norm2(fix)))

        # (0, 4, 1, 2, 3) channel_first
        shape2 = (0, self.ndim + 1, *list(range(1, self.ndim + 1)))
        mov = mov.permute(shape2)
        fix = fix.permute(shape2)
        return mov, fix


class Dilatestage(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        ndim,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilation,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cpe_per_satge=False,
        cpe_per_block=False,
        downsample=True,
        merging_way=None,
        dilate_attention=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ndim = ndim

        # build blocks
        self.blocks = nn.ModuleList([
            DilateBlock(
                ndim=ndim,
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                kernel_size=kernel_size,
                dilation=dilation,
                cpe_per_block=cpe_per_block,
                dilate_attention=dilate_attention,
            ) for i in range(depth)
        ])

        # patch merging layer
        self.downsample = (PatchMerging(ndim, dim, int(
            dim *
            2), merging_way, cpe_per_satge) if downsample else nn.Identity())

    def forward(self, x: Tensor):
        # B, C, [D], H, W
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        # (0, 2, 3, 4, 1) chanel_last
        shape = (0, *list(range(2, self.ndim + 2)), 1)

        mov, fix = x
        ori_mov = mov.permute(shape)
        ori_fix = fix.permute(shape)
        mov = self.downsample(mov)
        fix = self.downsample(fix)
        return (ori_mov, ori_fix), (mov, fix)


class DilateFormer(nn.Module):

    def __init__(
            self,
            ndim,
            img_size,
            in_chans,
            embed_dim,
            patch_size=4,
            depths=[2, 2, 6, 2],
            num_heads=[6, 12, 24, 48],
            dilation=[1, 2, 3],
            kernel_size=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            patch_way="overlaping",
            merging_way="conv3_2",
            dilate_attention=[True, True, False, False],
            downsamples=[True, True, True, False],
            cpe_per_satge=False,
            cpe_per_block=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
    ):
        super().__init__()
        self.ndim = ndim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.num_layers = len(depths)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # patch embedding
        self.patch_embed = PatchEmbed(
            ndim=ndim,
            img_size=img_size,
            in_chans=in_chans,
            hidden_dim=16,
            patch_size=patch_size,
            embed_dim=embed_dim,
            patch_way=patch_way,
        )

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # build layers
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage = Dilatestage(
                ndim=ndim,
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                kernel_size=kernel_size,
                dilation=dilation,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                act_layer=nn.GELU,
                norm_layer=norm_layer,
                cpe_per_satge=cpe_per_satge,
                cpe_per_block=cpe_per_block,
                downsample=downsamples[i_layer],
                merging_way=merging_way,
                dilate_attention=dilate_attention[i_layer],
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(stage)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"m_norm{i_layer}"
            self.add_module(layer_name, layer)
            layer = norm_layer(num_features[i_layer])
            layer_name = f"f_norm{i_layer}"
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        # self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DilateFormer, self).train(mode)
        self._freeze_stages()

    def forward(self, mov, fix):
        """Forward function."""
        mov = self.patch_embed(mov)
        fix = self.patch_embed(fix)
        x = (mov, fix)  # B, C, [D], H, W

        # (0, 4, 1, 2, 3) channel_first
        shape = (0, self.ndim + 1, *list(range(1, self.ndim + 1)))
        outs = []
        for i, stage in enumerate(self.stages):
            ori_x, x = stage(x)
            if i in self.out_indices:
                m_norm_layer = getattr(self, f"m_norm{i}")
                f_norm_layer = getattr(self, f"f_norm{i}")
                mov_out, fix_out = ori_x
                mov_out = m_norm_layer(mov_out)
                fix_out = f_norm_layer(fix_out)
                mov_out = mov_out.permute(shape).contiguous()
                fix_out = fix_out.permute(shape).contiguous()
                out = (mov_out, fix_out)
                outs.append(out)
        return outs


class ConvReLU(nn.Sequential):

    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = getattr(nn, f'Conv{ndim}d')(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = getattr(nn, f'InstanceNorm{ndim}d')(out_channels)
        else:
            nm = getattr(nn, f'BatchNorm{ndim}d')(out_channels)
        super(ConvReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
        skip2=True,
    ):
        super().__init__()
        if skip2:
            in_chnas = in_channels + skip_channels + skip_channels
        else:
            in_chnas = in_channels + skip_channels

        if ndim == 2:
            mode = 'bilinear'
        elif ndim == 3:
            mode = 'trilinear'

        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.conv1 = ConvReLU(
            ndim,
            in_chnas,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = ConvReLU(
            ndim,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip, skip2=None):
        x = self.up(x)
        if skip2 is not None:
            x = torch.cat([x, skip, skip2], dim=1)
        else:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


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


class ProjectionLayer(nn.Module):

    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.init_weights()

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat

    def init_weights(self):
        normal_init(self.proj, mean=0, std=1e-5, bias=0)


class CWM(nn.Module):

    def __init__(self, in_channels, channels):
        super(CWM, self).__init__()
        self.num_fields = in_channels // 3

        self.conv = nn.Sequential(
            ConvReLU(3, in_channels, channels, 3, 1, use_batchnorm=False),
            ConvReLU(3, channels, channels, 3, 1, use_batchnorm=False),
            nn.Conv3d(channels, self.num_fields, 3, 1, 1),
            nn.Softmax(dim=1),
        )
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='trilinear',
                                    align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        weight = self.conv(x)

        weighted_field = 0
        for i in range(self.num_fields):
            w = x[:, 3 * i:3 * (i + 1)]
            weight_map = weight[:, i:(i + 1)]
            weighted_field = weighted_field + w * weight_map

        return 2 * weighted_field


class ModeTransformer(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 kernel_size=3,
                 qk_scale=None,
                 use_rpb=True):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = kernel_size
        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb = nn.Parameter(
                torch.zeros(self.num_heads, self.rpb_size, self.rpb_size,
                            self.rpb_size))
        vectors = [
            torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3
        ]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        self.register_buffer('grid', grid, persistent=False)

    def makeV(self, N):
        # v.shape: (1, N, self.num_heads, self.kernel_size**3, 3)
        v = self.grid.reshape(self.kernel_size**3,
                              3).unsqueeze(0).unsqueeze(0).repeat(
                                  N, self.num_heads, 1, 1).unsqueeze(0)
        return v

    def apply_pb(self, attn, N):
        # attn: B, N, self.num_heads, 1, tokens = (3x3x3)
        bias_idx = torch.arange(self.rpb_size**3).unsqueeze(-1).repeat(N, 1)
        return attn + self.rpb.flatten(1, 3)[:, bias_idx].reshape(
            self.num_heads, N, 1, self.rpb_size**3).transpose(0, 1)

    def forward(self, q, k):

        B, H, W, T, C = q.shape
        N = H * W * T
        num_tokens = int(self.kernel_size**3)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(
            3, 4) * self.scale  # 1, N, heads, 1, head_dim
        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1

        k = k.permute(0, 4, 1, 2, 3)  # C, H, W, T
        k = F.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        k = k.flatten(0, 1)  # C, H+2, W+2, T+2
        k = k.unfold(1, self.kernel_size,
                     1).unfold(2, self.kernel_size,
                               1).unfold(3, self.kernel_size,
                                         1).permute(0, 4, 5, 6, 1, 2,
                                                    3)  # C, 3, 3, 3, H, W, T
        k = k.reshape(B, self.num_heads, C // self.num_heads, num_tokens,
                      N)  # memory boom
        k = k.permute(0, 4, 1, 3, 2)  # (B, N, heads, num_tokens, head_dim)

        attn = (q @ k.transpose(-2, -1))  # =>B x N x heads x 1 x num_tokens
        if self.use_rpb:
            attn = self.apply_pb(attn, N)
        attn = attn.softmax(dim=-1)

        v = self.makeV(N)  # B, N, heads, num_tokens, 3
        x = (attn @ v)  # B x N x heads x 1 x 3
        x = x.reshape(B, H, W, T, self.num_heads * 3).permute(0, 4, 1, 2, 3)

        return x


####################################
class WarpII(nn.Module):
    """Warp an image with given flow / dense displacement field (DDF).

    Args:
        normalization (bool, optional): Normalize flow field. Default: ``False``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``True``
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
    """

    def __init__(
        self,
        normalization: bool = False,
        align_corners: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.default_interp_mode = 'bilinear'

    @staticmethod
    def create_grid(size, normalization=False) -> Tensor:
        if normalization:
            vectors = [torch.linspace(-1, 1, s) for s in size]
        else:
            vectors = [torch.arange(0, s) for s in size]
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            grids = torch.meshgrid(vectors, indexing='ij')
        else:
            grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).unsqueeze_(0)
        grid = grid.type(torch.FloatTensor)
        return grid

    def forward(self, flow: Tensor, image: Tensor, interp_mode=None) -> Tensor:
        """
        Warp image with flow.
        Args:
            flow (Tensor): flow field of shape [batch_size, spatial_dims, ...]
            image (Tensor): input image of shape [batch_size, channels, ...]
            interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]

        Returns:
            Tensor: Warped image.
        """

        if interp_mode is None:
            interp_mode = self.default_interp_mode

        img_size = flow.shape[2:]
        ndim = len(img_size)

        # create sampling grid
        grid = self.create_grid(img_size, self.normalization).to(flow)

        # warped deformation filed
        warped_grid = grid + flow

        # normalize grid values to [-1, 1] for resampler
        if not self.normalization:
            scale = torch.tensor([s - 1 for s in img_size])
            scale = scale.view((1, ndim, 1, 1, 1)).to(warped_grid)
            warped_grid = 2 * warped_grid / scale - 1.0

        # move channels dim to last position also not sure why,
        # but the channels need to be reversed
        warped_grid = warped_grid.permute(
            (0, *[i + 2 for i in range(ndim)], 1))
        warped_grid = warped_grid.flip(-1)

        return F.grid_sample(
            image,
            warped_grid,
            mode=interp_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class ResizeTransform(nn.Module):

    def __init__(self, scale_factor, mode='trilinear'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x,
                              align_corners=True,
                              scale_factor=self.factor,
                              mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x,
                              align_corners=True,
                              scale_factor=self.factor,
                              mode=self.mode)

        return x


class PixelShuffle3d(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale**3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale,
                                             self.scale, self.scale, in_depth,
                                             in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class UpsampleBlock(nn.Module):

    def __init__(self, scale_factor, in_dim, kernel_size, grouping=False):
        super().__init__()
        self.in_dim = in_dim
        self.scale_factor = scale_factor
        groups = in_dim if grouping else 1
        self.conv = nn.Conv3d(in_dim, (scale_factor**3) * in_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              groups=groups)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = PixelShuffle3d(self.scale_factor)(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class CoarseToFineDecoderBlock(nn.Module):

    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        feat_channels=0,
        use_batchnorm=False,
        scale_factor=2,
    ):
        super().__init__()
        self.up = UpsampleBlock(scale_factor, in_channels, kernel_size=3)
        # if ndim == 2:
        #     mode = 'bilinear'
        # elif ndim == 3:
        #     mode = 'trilinear'
        # self.up = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=False)
        self.conv1 = ConvReLU(
            ndim,
            in_channels * 2 + feat_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = ConvReLU(
            ndim,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, mov, fix, feat=None):
        if feat is not None:
            feat = self.up(feat)
            x = torch.cat([mov, feat, fix], dim=1)
        else:
            x = torch.cat([mov, fix], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CoarseToFineDecoder(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        ndim = 3
        # self.decoder_4 = CoarseToFineDecoderBlock(
        #     ndim=ndim,
        #     in_channels=embed_dim * 4,
        #     out_channels=embed_dim * 2,
        #     feat_channels=0)
        self.decoder_3 = CoarseToFineDecoderBlock(ndim=3,
                                                  in_channels=embed_dim * 4,
                                                  out_channels=embed_dim * 2,
                                                  feat_channels=0)
        self.decoder_2 = CoarseToFineDecoderBlock(ndim=3,
                                                  in_channels=embed_dim * 2,
                                                  out_channels=embed_dim,
                                                  feat_channels=embed_dim * 2)
        self.decoder_1 = CoarseToFineDecoderBlock(ndim=3,
                                                  in_channels=embed_dim,
                                                  out_channels=embed_dim // 2,
                                                  feat_channels=embed_dim)
        self.decoder_0 = CoarseToFineDecoderBlock(ndim=3,
                                                  in_channels=embed_dim // 2,
                                                  out_channels=embed_dim // 4,
                                                  feat_channels=embed_dim // 2,
                                                  scale_factor=2)

        # self.flow_4 = DefaultFlow(embed_dim * 2, ndim)
        self.flow_3 = DefaultFlow(embed_dim * 2, ndim)
        self.flow_2 = DefaultFlow(embed_dim, ndim)
        self.flow_1 = DefaultFlow(embed_dim // 2, ndim)

        self.warp = WarpII(normalization=False, align_corners=True)
        self.resize = ResizeTransform(scale_factor=2, mode='trilinear')
        # self.resize_x4 = ResizeTransform(scale_factor=4, mode='trilinear')
        self.up = UpsampleBlock(2, in_dim=embed_dim // 4, kernel_size=3)

    def forward(self, feats):
        mov_0, fix_0 = feats[0]
        mov_1, fix_1 = feats[1]
        mov_2, fix_2 = feats[2]
        mov_3, fix_3 = feats[3]
        # mov_4, fix_4 = feats[4]

        # Step 4
        # x = self.decoder_4(mov_4, fix_4)
        # flow_4 = self.flow_4(x)

        # # flow_4 = flow_4 + flow_init
        # flow_4_up = self.resize(flow_4)
        # mov_3 = self.warp(flow_4_up, mov_3)

        # Step 3
        # x = self.decoder_3(mov_3, fix_3, x)
        x = self.decoder_3(mov_3, fix_3)
        flow_3 = self.flow_3(x)

        # flow_3 = flow_3 + flow_4_up
        flow_3_up = self.resize(flow_3)
        mov_2 = self.warp(flow_3_up, mov_2)

        # Step 2
        x = self.decoder_2(mov_2, fix_2, x)
        flow_2 = self.flow_2(x)

        flow_2 = flow_2 + flow_3_up
        flow_2_up = self.resize(flow_2)
        mov_1 = self.warp(flow_2_up, mov_1)

        # Step 1
        x = self.decoder_1(mov_1, fix_1, x)
        flow_1 = self.flow_1(x)

        flow_1 = flow_1 + flow_2_up
        flow_1_up = self.resize(flow_1)
        mov_0 = self.warp(flow_1_up, mov_0)

        # Step 0
        x = self.decoder_0(mov_0, fix_0, x)
        x = self.up(x)
        return x
