# Copyright (c) MMIPT. All rights reserved.
import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

# from mmipt.registry import MODELS
from .modulev2 import (CoarseToFineDecoder, ConvReLU, DecoderBlock,
                       DefaultFlow, DilateFormer, Warp)
from .mind_utils import mind_ssc


#@MODELS.register_module()
class DilateMorph(BaseModule):

    def __init__(
        self,
        img_size,
        in_chans=1,
        embed_dim=96,
        patch_size=4,
        depths=(2, 2, 4, 2),
        num_heads=(3, 6, 12, 24),
        dilation=(1, 2, 3),
        kernel_size=3,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.2,
        patch_way="overlaping",
        merging_way="conv3_2",
        dilate_attention=[True, True, False, False],
        downsamples=[True, True, True, False],
        cpe_per_satge=False,
        cpe_per_block=True,
        out_indices=(0, 1, 2, 3),
        if_convskip=True,
        if_transskip=True,
        frozen_stages=-1,
        use_checkpoint=False,
        init_cfg=None,
        ndim=3,
        use_mind_ssc=False,
    ):
        super().__init__(init_cfg=init_cfg)
        self.if_convskip = if_convskip
        self.if_transskip = if_transskip
        mode = "trilinear" if ndim == 3 else 'bilinear'
        if use_mind_ssc:
            in_chans = 12
            self.use_mind_ssc = True

        if self.if_convskip:
            self.avg_pool = getattr(nn, f'AvgPool{ndim}d')(3,
                                                           stride=2,
                                                           padding=1)
            self.c1 = ConvReLU(ndim,
                               2 * in_chans,
                               embed_dim // 2,
                               3,
                               1,
                               use_batchnorm=False)

        self.transformer = DilateFormer(
            ndim=ndim,
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            dilation=dilation,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            # norm_layer=partial(nn.LayerNorm, eps=1e-6),
            patch_way=patch_way,
            merging_way=merging_way,
            dilate_attention=dilate_attention,
            downsamples=downsamples,
            cpe_per_satge=cpe_per_satge,
            cpe_per_block=cpe_per_block,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint)
        self.up0 = DecoderBlock(ndim,
                                embed_dim * 8,
                                embed_dim * 4,
                                skip_channels=embed_dim *
                                4 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(ndim,
                                embed_dim * 4,
                                embed_dim * 2,
                                skip_channels=embed_dim *
                                2 if if_transskip else 0,
                                use_batchnorm=False)
        self.up2 = DecoderBlock(ndim,
                                embed_dim * 2,
                                embed_dim,
                                skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)
        self.up3 = DecoderBlock(ndim,
                                embed_dim,
                                embed_dim // 2,
                                skip_channels=embed_dim //
                                2 if if_convskip else 0,
                                use_batchnorm=False,
                                skip2=False)
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.flow = DefaultFlow(embed_dim // 2)
        self.warp = Warp(img_size)

    def forward(self, source: Tensor, target: Tensor, **kwargs):
        if getattr(self, 'use_mind_ssc', False):
            source = mind_ssc(source)
            target = mind_ssc(target)

        if self.if_convskip:
            x = torch.cat((source, target), dim=1)
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None

        out = self.transformer(source, target)
        if self.if_transskip:
            mov_f1, fix_f1 = out[-2]
            mov_f2, fix_f2 = out[-3]
            mov_f3, fix_f3 = out[-4]
        else:
            mov_f1, fix_f1 = None, None
            mov_f2, fix_f2 = None, None
            mov_f3, fix_f3 = None, None

        mov_f0, fix_f0 = out[-1]
        # x = self.up0(mov_f0 + fix_f0, mov_f1, fix_f1)
        x = self.up0(mov_f0, mov_f1, fix_f1)
        x = self.up1(x, mov_f2, fix_f2)
        x = self.up2(x, mov_f3, fix_f3)
        x = self.up3(x, f4)
        x = self.up(x)
        flow = self.flow(x)
        warped = self.warp(source, flow=flow)
        return warped, flow


class DilateMorph1(BaseModule):

    def __init__(
        self,
        img_size,
        in_chans=1,
        embed_dim=96,
        patch_size=4,
        depths=(2, 2, 4, 2),
        num_heads=(3, 6, 12, 24),
        dilation=(1, 2, 3),
        kernel_size=3,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.2,
        patch_way="overlaping",
        merging_way="conv3_2",
        dilate_attention=[True, True, False, False],
        downsamples=[True, True, True, False],
        cpe_per_satge=False,
        cpe_per_block=True,
        out_indices=(0, 1, 2, 3),
        if_convskip=True,
        if_transskip=True,
        frozen_stages=-1,
        use_checkpoint=False,
        init_cfg=None,
        ndim=3,
    ):
        super().__init__(init_cfg=init_cfg)
        self.if_convskip = if_convskip
        self.if_transskip = if_transskip
        mode = "trilinear" if ndim == 3 else 'bilinear'

        if self.if_convskip:
            self.avg_pool = getattr(nn, f'AvgPool{ndim}d')(3,
                                                           stride=2,
                                                           padding=1)
            self.c1 = ConvReLU(ndim,
                               in_chans,
                               embed_dim // 2,
                               3,
                               1,
                               use_batchnorm=False)

        self.transformer = DilateFormer(
            ndim=ndim,
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            dilation=dilation,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            # norm_layer=partial(nn.LayerNorm, eps=1e-6),
            patch_way=patch_way,
            merging_way=merging_way,
            dilate_attention=dilate_attention,
            downsamples=downsamples,
            cpe_per_satge=cpe_per_satge,
            cpe_per_block=cpe_per_block,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint)
        self.up0 = DecoderBlock(ndim,
                                embed_dim * 8,
                                embed_dim * 4,
                                skip_channels=embed_dim *
                                4 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(ndim,
                                embed_dim * 4,
                                embed_dim * 2,
                                skip_channels=embed_dim *
                                2 if if_transskip else 0,
                                use_batchnorm=False)
        self.up2 = DecoderBlock(ndim,
                                embed_dim * 2,
                                embed_dim,
                                skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)
        self.up3 = DecoderBlock(ndim,
                                embed_dim,
                                embed_dim // 2,
                                skip_channels=embed_dim //
                                2 if if_convskip else 0,
                                use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.flow = DefaultFlow(embed_dim // 2)
        self.warp = Warp(img_size)

    def forward(self, source: Tensor, target: Tensor, **kwargs):
        if self.if_convskip:
            mov_f4 = self.c1(self.avg_pool(source))
            fix_f4 = self.c1(self.avg_pool(target))
        else:
            mov_f4 = None
            fix_f4 = None

        out = self.transformer(source, target)
        if self.if_transskip:
            mov_f1, fix_f1 = out[-2]
            mov_f2, fix_f2 = out[-3]
            mov_f3, fix_f3 = out[-4]
        else:
            mov_f1, fix_f1 = None, None
            mov_f2, fix_f2 = None, None
            mov_f3, fix_f3 = None, None

        x = out[-1][0] + out[-1][1]
        x = self.up0(x, mov_f1, fix_f1)
        x = self.up1(x, mov_f2, fix_f2)
        x = self.up2(x, mov_f3, fix_f3)
        x = self.up3(x, mov_f4, fix_f4)
        x = self.up(x)
        flow = self.flow(x)
        warped = self.warp(source, flow=flow)
        return warped, flow


#@MODELS.register_module()
class DilateMorphHalf(BaseModule):

    def __init__(
        self,
        img_size,
        in_chans=1,
        embed_dim=96,
        flow_dim=16,
        patch_size=4,
        depths=(2, 2, 4, 2),
        num_heads=(3, 6, 12, 24),
        dilation=(1, 2, 3),
        kernel_size=3,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.2,
        patch_way="overlaping",
        merging_way="conv3_2",
        dilate_attention=[True, True, False, False],
        downsamples=[True, True, True, False],
        cpe_per_satge=False,
        cpe_per_block=True,
        out_indices=(0, 1, 2, 3),
        if_convskip=True,
        if_transskip=True,
        frozen_stages=-1,
        use_checkpoint=False,
        init_cfg=None,
        ndim=3,
        use_bottleneck=True,
    ):
        super().__init__(init_cfg=init_cfg)
        self.if_convskip = if_convskip
        self.if_transskip = if_transskip

        if self.if_convskip:
            self.avg_pool = getattr(nn, f'AvgPool{ndim}d')(3,
                                                           stride=2,
                                                           padding=1)
            self.c1 = ConvReLU(ndim,
                               2 * in_chans,
                               embed_dim // 2,
                               3,
                               1,
                               use_batchnorm=False)

        self.transformer = DilateFormer(
            ndim=ndim,
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            dilation=dilation,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            # norm_layer=partial(nn.LayerNorm, eps=1e-6),
            patch_way=patch_way,
            merging_way=merging_way,
            dilate_attention=dilate_attention,
            downsamples=downsamples,
            cpe_per_satge=cpe_per_satge,
            cpe_per_block=cpe_per_block,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint)
        if use_bottleneck:
            self.bottleneck = ConvReLU(ndim,
                                       embed_dim * 16,
                                       embed_dim * 8,
                                       3,
                                       1,
                                       1,
                                       use_batchnorm=False)
        self.up0 = DecoderBlock(ndim,
                                embed_dim * 8,
                                embed_dim * 4,
                                skip_channels=embed_dim *
                                4 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(ndim,
                                embed_dim * 4,
                                embed_dim * 2,
                                skip_channels=embed_dim *
                                2 if if_transskip else 0,
                                use_batchnorm=False)
        self.up2 = DecoderBlock(ndim,
                                embed_dim * 2,
                                embed_dim,
                                skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)
        self.up3 = DecoderBlock(ndim,
                                embed_dim,
                                flow_dim,
                                skip_channels=embed_dim //
                                2 if if_convskip else 0,
                                use_batchnorm=False,
                                skip2=False)

    def forward(self, source: Tensor, target: Tensor, **kwargs):
        if self.if_convskip:
            x = torch.cat((source, target), dim=1)
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None

        out = self.transformer(source, target)
        if self.if_transskip:
            mov_f1, fix_f1 = out[-2]
            mov_f2, fix_f2 = out[-3]
            mov_f3, fix_f3 = out[-4]
        else:
            mov_f1, fix_f1 = None, None
            mov_f2, fix_f2 = None, None
            mov_f3, fix_f3 = None, None

        x = self.bottleneck(torch.cat(out[-1], dim=1))
        x = self.up0(x, mov_f1, fix_f1)
        x = self.up1(x, mov_f2, fix_f2)
        x = self.up2(x, mov_f3, fix_f3)
        x = self.up3(x, f4)
        return x


#@MODELS.register_module()
class DilateMorphBi(DilateMorph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                source: Tensor,
                target: Tensor,
                training: bool = False,
                **kwargs):
        if self.if_convskip:
            mov_f4 = torch.cat((source, target), dim=1)
            mov_f4 = self.avg_pool(mov_f4)
            mov_f4 = self.c1(mov_f4)
            if training:
                fix_f4 = torch.cat((target, source), dim=1)
                fix_f4 = self.avg_pool(fix_f4)
                fix_f4 = self.c1(fix_f4)
        else:
            mov_f4 = None
            fix_f4 = None

        out = self.transformer(source, target)
        if self.if_transskip:
            mov_f1, fix_f1 = out[-2]
            mov_f2, fix_f2 = out[-3]
            mov_f3, fix_f3 = out[-4]
        else:
            mov_f1, fix_f1 = None, None
            mov_f2, fix_f2 = None, None
            mov_f3, fix_f3 = None, None

        mov_f0, fix_f0 = out[-1]
        mx = self.up0(mov_f0, mov_f1, fix_f1)
        mx = self.up1(mx, mov_f2, fix_f2)
        mx = self.up2(mx, mov_f3, fix_f3)
        mx = self.up3(mx, mov_f4)
        mx = self.up(mx)

        mov_flow = self.flow(mx)
        warped_mov = self.warp(source, flow=mov_flow)

        if training:
            fx = self.up0(fix_f0, fix_f1, mov_f1)
            fx = self.up1(fx, fix_f2, mov_f2)
            fx = self.up2(fx, fix_f3, mov_f3)
            fx = self.up3(fx, fix_f4)
            fx = self.up(fx)
            fix_flow = self.flow(fx)
            warped_fix = self.warp(target, flow=fix_flow)
            return warped_mov, warped_fix, mov_flow, fix_flow

        return warped_mov, mov_flow


#@MODELS.register_module()
class DilateMorphHalfBi(DilateMorphHalf):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_bottleneck=False)

    def forward(self,
                source: Tensor,
                target: Tensor,
                training: bool = False,
                **kwargs):
        if self.if_convskip:
            mov_f4 = torch.cat((source, target), dim=1)
            mov_f4 = self.avg_pool(mov_f4)
            mov_f4 = self.c1(mov_f4)
            if training:
                fix_f4 = torch.cat((target, source), dim=1)
                fix_f4 = self.avg_pool(fix_f4)
                fix_f4 = self.c1(fix_f4)
        else:
            mov_f4 = None
            fix_f4 = None

        out = self.transformer(source, target)
        if self.if_transskip:
            mov_f1, fix_f1 = out[-2]
            mov_f2, fix_f2 = out[-3]
            mov_f3, fix_f3 = out[-4]
        else:
            mov_f1, fix_f1 = None, None
            mov_f2, fix_f2 = None, None
            mov_f3, fix_f3 = None, None

        mov_f0, fix_f0 = out[-1]
        mx = self.up0(mov_f0, mov_f1, fix_f1)
        mx = self.up1(mx, mov_f2, fix_f2)
        mx = self.up2(mx, mov_f3, fix_f3)
        mx = self.up3(mx, mov_f4)

        if training:
            fx = self.up0(fix_f0, fix_f1, mov_f1)
            fx = self.up1(fx, fix_f2, mov_f2)
            fx = self.up2(fx, fix_f3, mov_f3)
            fx = self.up3(fx, fix_f4)
            return dict(source=mx, target=fx)

        return mx


#@MODELS.register_module()
class DilateMorphCascadeAd(nn.Module):

    def __init__(
        self,
        img_size,
        in_chans=1,
        embed_dim=96,
        flow_dim=16,
        patch_size=4,
        depths=(2, 2, 4, 2),
        num_heads=(3, 6, 12, 24),
        dilation=(1, 2, 3),
        kernel_size=3,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.2,
        patch_way="overlaping",
        merging_way="conv3_2",
        dilate_attention=[True, True, False, False],
        downsamples=[True, True, True, False],
        cpe_per_satge=False,
        cpe_per_block=True,
        out_indices=(0, 1, 2, 3),
        if_convskip=True,
        if_transskip=True,
        frozen_stages=-1,
        use_checkpoint=False,
        init_cfg=None,
        ndim=3,
    ):
        super().__init__(init_cfg=init_cfg)
        self.if_convskip = if_convskip
        self.if_transskip = if_transskip

        if self.if_convskip:
            self.avg_pool = getattr(nn, f'AvgPool{ndim}d')(3,
                                                           stride=2,
                                                           padding=1)
            self.c1 = ConvReLU(ndim,
                               2 * in_chans,
                               embed_dim // 2,
                               3,
                               1,
                               use_batchnorm=False)

        self.transformer = DilateFormer(
            ndim=ndim,
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            dilation=dilation,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            # norm_layer=partial(nn.LayerNorm, eps=1e-6),
            patch_way=patch_way,
            merging_way=merging_way,
            dilate_attention=dilate_attention,
            downsamples=downsamples,
            cpe_per_satge=cpe_per_satge,
            cpe_per_block=cpe_per_block,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint)

        self.up0 = DecoderBlock(ndim,
                                embed_dim * 8,
                                embed_dim * 4,
                                skip_channels=embed_dim *
                                4 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(ndim,
                                embed_dim * 4,
                                embed_dim * 2,
                                skip_channels=embed_dim *
                                2 if if_transskip else 0,
                                use_batchnorm=False)
        self.up2 = DecoderBlock(ndim,
                                embed_dim * 2,
                                embed_dim,
                                skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)
        self.up3 = DecoderBlock(ndim,
                                embed_dim,
                                flow_dim,
                                skip_channels=embed_dim //
                                2 if if_convskip else 0,
                                use_batchnorm=False,
                                skip2=False)

    def forward(self, source: Tensor, target: Tensor, **kwargs):
        if self.if_convskip:
            x = torch.cat((source, target), dim=1)
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None

        out = self.transformer(source, target)
        if self.if_transskip:
            mov_f1, fix_f1 = out[-2]
            mov_f2, fix_f2 = out[-3]
            mov_f3, fix_f3 = out[-4]
        else:
            mov_f1, fix_f1 = None, None
            mov_f2, fix_f2 = None, None
            mov_f3, fix_f3 = None, None

        x = self.bottleneck(torch.cat(out[-1], dim=1))
        x = self.up0(x, mov_f1, fix_f1)
        x = self.up1(x, mov_f2, fix_f2)
        x = self.up2(x, mov_f3, fix_f3)
        x = self.up3(x, f4)
        return x


# @MODELS.register_module()
class DilateMorphCoarseToFine(BaseModule):

    def __init__(
        self,
        img_size,
        in_chans=1,
        embed_dim=96,
        patch_size=4,
        depths=(2, 2, 4, 2),
        num_heads=(3, 6, 12, 24),
        dilation=(1, 2, 3),
        kernel_size=3,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.2,
        patch_way="overlaping",
        merging_way="conv3_2",
        dilate_attention=[True, True, False, False],
        downsamples=[True, True, True, False],
        cpe_per_satge=False,
        cpe_per_block=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        init_cfg=None,
        ndim=3,
    ):
        super().__init__(init_cfg=init_cfg)
        self.avg_pool = getattr(nn, f'AvgPool{ndim}d')(3, stride=2, padding=1)
        self.mov_c4 = ConvReLU(ndim,
                               in_channels=in_chans,
                               out_channels=embed_dim // 2,
                               kernel_size=3,
                               padding=1,
                               use_batchnorm=False)
        self.fix_c4 = ConvReLU(ndim,
                               in_channels=in_chans,
                               out_channels=embed_dim // 2,
                               kernel_size=3,
                               padding=1,
                               use_batchnorm=False)
        self.transformer = DilateFormer(ndim=ndim,
                                        img_size=img_size,
                                        in_chans=in_chans,
                                        embed_dim=embed_dim,
                                        patch_size=patch_size,
                                        depths=depths,
                                        num_heads=num_heads,
                                        dilation=dilation,
                                        kernel_size=kernel_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop_rate=drop_rate,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=drop_path_rate,
                                        patch_way=patch_way,
                                        merging_way=merging_way,
                                        dilate_attention=dilate_attention,
                                        downsamples=downsamples,
                                        cpe_per_satge=cpe_per_satge,
                                        cpe_per_block=cpe_per_block,
                                        out_indices=out_indices,
                                        frozen_stages=frozen_stages,
                                        use_checkpoint=use_checkpoint)
        self.decoder = CoarseToFineDecoder(embed_dim)
        self.flow = DefaultFlow(embed_dim // 4)
        self.warp = Warp(img_size)

    def forward(self, source: Tensor, target: Tensor, **kwargs):
        out = self.transformer(source, target)
        out.insert(0, (
            self.mov_c4(self.avg_pool(source)),
            self.fix_c4(self.avg_pool(target)),
        ))
        x = self.decoder(out)
        flow = self.flow(x)
        warped = self.warp(source, flow=flow)
        return warped, flow
