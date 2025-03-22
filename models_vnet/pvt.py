import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmengine.model.weight_init import constant_init, trunc_normal_init
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

from ..models_transmorph.transmorph import Conv3dReLU, DecoderBlock
from .model_util import Warp, DefaultFlow


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
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


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                dim,
                dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, L):
        B, N, C = x.shape
        q = self.q(x).reshape(
            B,
            N,
            self.num_heads,
            C // self.num_heads,
        ).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W, L)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(
                B,
                -1,
                2,
                self.num_heads,
                C // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(
                B,
                -1,
                2,
                self.num_heads,
                C // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, H, W, L):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, L))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = _triple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.L = img_size[2] // patch_size[2]
        self.num_patches = self.H * self.W * self.L
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W, L = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H = H // self.patch_size[0]
        W = W // self.patch_size[1]
        L = L // self.patch_size[2]
        return x, (H, W, L)


class PyramidVisionTransformer(nn.Module):

    def __init__(self,
                 img_size=(160, 192, 224),
                 patch_size=16,
                 in_chans=2,
                 num_classes=1000,
                 embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3),
                 sr_ratios=(8, 4, 2, 1),
                 F4=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.F4 = F4

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=(img_size[0] // 4,
                                                 img_size[1] // 4,
                                                 img_size[2] // 4),
                                       patch_size=2,
                                       in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=(img_size[0] // 8,
                                                 img_size[1] // 8,
                                                 img_size[2] // 8),
                                       patch_size=2,
                                       in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=(img_size[0] // 16,
                                                 img_size[1] // 16,
                                                 img_size[2] // 16),
                                       patch_size=2,
                                       in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        # pos_embed
        self.pos_embed1 = nn.Parameter(
            torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(
            torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(
            torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(
            torch.zeros(1, self.patch_embed4.num_patches + 1, embed_dims[3]))
        self.pos_drop4 = nn.Dropout(p=drop_rate)

        # transformer encoder
        # stochastic depth decay rule
        dpr = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, dpr, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0],
                  num_heads=num_heads[0],
                  mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_layer=norm_layer,
                  sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1],
                  num_heads=num_heads[1],
                  mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_layer=norm_layer,
                  sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2],
                  num_heads=num_heads[2],
                  mlp_ratio=mlp_ratios[2],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_layer=norm_layer,
                  sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3],
                  num_heads=num_heads[3],
                  mlp_ratio=mlp_ratios[3],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_layer=norm_layer,
                  sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])

        # init weights
        nn.init.trunc_normal_(self.pos_embed1, std=.02)
        nn.init.trunc_normal_(self.pos_embed2, std=.02)
        nn.init.trunc_normal_(self.pos_embed3, std=.02)
        nn.init.trunc_normal_(self.pos_embed4, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=0.02, bias=0.0)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W, L):
        if H * W * L == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(
                    1,
                    patch_embed.H,
                    patch_embed.W,
                    patch_embed.L,
                    -1,
                ).permute(0, 4, 1, 2, 3),
                size=(H, W, L),
                mode="trilinear",
            ).reshape(1, -1, H * W * L).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        # stage 1
        x, (H, W, L) = self.patch_embed1(x)
        pos_embed1 = self._get_pos_embed(
            self.pos_embed1,
            self.patch_embed1,
            H,
            W,
            L,
        )
        x = x + pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W, L)
        x = x.reshape(B, H, W, L, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 2
        x, (H, W, L) = self.patch_embed2(x)
        pos_embed2 = self._get_pos_embed(
            self.pos_embed2,
            self.patch_embed2,
            H,
            W,
            L,
        )
        x = x + pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W, L)
        x = x.reshape(B, H, W, L, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 3
        x, (H, W, L) = self.patch_embed3(x)
        pos_embed3 = self._get_pos_embed(
            self.pos_embed3,
            self.patch_embed3,
            H,
            W,
            L,
        )
        x = x + pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W, L)
        x = x.reshape(B, H, W, L, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 4
        x, (H, W, L) = self.patch_embed4(x)
        pos_embed4 = self._get_pos_embed(
            self.pos_embed4[:, 1:],
            self.patch_embed4,
            H,
            W,
            L,
        )
        x = x + pos_embed4
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W, L)
        x = x.reshape(B, H, W, L, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.F4:
            x = x[3:4]
        return x


class PVTVNet(nn.Module):

    def __init__(
        self,
        img_size,
        patch_size=4,
        embed_dims=(20, 40, 200, 320),  # differ from original PVT
        num_heads=(2, 4, 8, 16),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        depths=(3, 10, 60, 3),  # differ from original PVT
        sr_ratios=(8, 4, 2, 1),
        drop_rate=0.0,
        drop_path_rate=0.1,
        flow_dim=16,
    ):
        super(PVTVNet, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU(2, flow_dim, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, flow_dim, 3, 1, use_batchnorm=False)

        self.transformer = PyramidVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=2,
            num_classes=1000,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop_rate=drop_rate,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            depths=depths,
            sr_ratios=sr_ratios,
            F4=False)
        self.up1 = DecoderBlock(embed_dims[-1],
                                embed_dims[-2],
                                skip_channels=embed_dims[-2],
                                use_batchnorm=False)
        self.up2 = DecoderBlock(embed_dims[-2],
                                embed_dims[-3],
                                skip_channels=embed_dims[-3],
                                use_batchnorm=False)
        self.up3 = DecoderBlock(embed_dims[-3],
                                embed_dims[-4],
                                skip_channels=embed_dims[-4],
                                use_batchnorm=False)
        self.up4 = DecoderBlock(embed_dims[-4],
                                16,
                                skip_channels=16,
                                use_batchnorm=False)
        self.up5 = DecoderBlock(16,
                                flow_dim,
                                skip_channels=flow_dim,
                                use_batchnorm=False)
        self.flow = DefaultFlow(16)
        self.warp = Warp(img_size)

    def forward(self, source, target, **kwargs):
        x = torch.cat([source, target], dim=1)

        x_s0 = x.clone()
        x_s1 = self.avg_pool(x)
        f4 = self.c1(x_s1)
        f5 = self.c2(x_s0)

        out = self.transformer(x)
        x = self.up1(out[-1], out[-2])
        x = self.up2(x, out[-3])
        x = self.up3(x, out[-4])
        x = self.up4(x, f4)
        x = self.up5(x, f5)
        flow = self.flow(x)

        warped = self.warp(source, flow=flow)
        return warped, flow
