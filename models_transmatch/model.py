# Copyright (c) MMIPT. All rights reserved.
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import build_conv_layer
from mmengine.model import normal_init
from . import decoder, lwca, lwsa


# @MODELS.register_module()
class TransMatch(nn.Module):

    def __init__(
            self,
            img_size,
            window_size,
            patch_size=4,
            in_chans=1,
            embed_dim=96,
            depths=(2, 2, 4, 2),
            num_heads=(4, 4, 8, 8),
            mlp_ratio=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            pat_merg_rf=4,
            out_indices=(0, 1, 2, 3),
    ):
        super().__init__()

        # Optional Convolution
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = decoder.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        # self.c2 = decoder.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        # LWSA
        self.backbone = lwsa.LWSA(patch_size=patch_size,
                                  in_chans=in_chans,
                                  embed_dim=embed_dim,
                                  depths=depths,
                                  num_heads=num_heads,
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  drop_rate=drop_rate,
                                  drop_path_rate=drop_path_rate,
                                  patch_norm=patch_norm,
                                  use_checkpoint=use_checkpoint,
                                  out_indices=out_indices,
                                  pat_merg_rf=pat_merg_rf)
        # self.moving_backbone = backbone
        # self.fixed_backbone = backbone

        # LWCA
        lwca_config = dict(patch_size=patch_size,
                           embed_dim=embed_dim,
                           depths=depths,
                           num_heads=num_heads,
                           window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           pat_merg_rf=pat_merg_rf,
                           qkv_bias=qkv_bias,
                           drop_rate=drop_rate,
                           drop_path_rate=drop_path_rate,
                           patch_norm=patch_norm,
                           use_checkpoint=use_checkpoint,
                           out_indices=out_indices)
        self.crossattn1 = lwca.LWCA(dim_diy=96, **lwca_config)
        self.crossattn2 = lwca.LWCA(dim_diy=192, **lwca_config)
        self.crossattn3 = lwca.LWCA(dim_diy=384, **lwca_config)
        self.crossattn4 = lwca.LWCA(dim_diy=768, **lwca_config)

        self.up0 = decoder.DecoderBlock(768,
                                        384,
                                        skip_channels=384,
                                        use_batchnorm=False)
        self.up1 = decoder.DecoderBlock(384,
                                        192,
                                        skip_channels=192,
                                        use_batchnorm=False)
        self.up2 = decoder.DecoderBlock(192,
                                        96,
                                        skip_channels=96,
                                        use_batchnorm=False)
        self.up3 = decoder.DecoderBlock(96,
                                        48,
                                        skip_channels=48,
                                        use_batchnorm=False)
        self.up4 = decoder.DecoderBlock(48,
                                        16,
                                        skip_channels=16,
                                        use_batchnorm=False)

        self.up = nn.Upsample(scale_factor=2,
                              mode='trilinear',
                              align_corners=False)

        self.flow = DefaultFlow(48)

        self.warp = Warp(img_size)

    def forward(self, source, target, **kwargs):

        # Batch, channel, height, width, depth
        input_fusion = torch.cat((source, target), dim=1)

        x_s1 = self.avg_pool(input_fusion)
        f4 = self.c1(x_s1)
        # f5 = self.c2(input_fusion)

        mov_feat_4, mov_feat_8, mov_feat_16, mov_feat_32 = self.backbone(
            source)
        fix_feat_4, fix_feat_8, fix_feat_16, fix_feat_32 = self.backbone(
            target)

        # LWCA module
        mov_feat_4_cross = self.crossattn1(mov_feat_4, fix_feat_4)
        mov_feat_8_cross = self.crossattn2(mov_feat_8, fix_feat_8)
        mov_feat_16_cross = self.crossattn3(mov_feat_16, fix_feat_16)
        mov_feat_32_cross = self.crossattn4(mov_feat_32, fix_feat_32)

        fix_feat_4_cross = self.crossattn1(fix_feat_4, mov_feat_4)
        fix_feat_8_cross = self.crossattn2(fix_feat_8, mov_feat_8)
        fix_feat_16_cross = self.crossattn3(fix_feat_16, mov_feat_16)
        fix_feat_32_cross = self.crossattn4(fix_feat_32, mov_feat_32)

        # try concat mov_feat_32 and fix_feat_32
        x = self.up0(mov_feat_32_cross, mov_feat_16_cross, fix_feat_16_cross)
        x = self.up1(x, mov_feat_8_cross, fix_feat_8_cross)
        x = self.up2(x, mov_feat_4_cross, fix_feat_4_cross)
        x = self.up3(x, f4)
        x = self.up(x)
        flow = self.flow(x)

        warped = self.warp(source, flow=flow)
        return warped, flow


# @MODELS.register_module()
class TransMatchDual(nn.Module):

    def __init__(
            self,
            window_size,
            patch_size=4,
            in_chans=1,
            embed_dim=96,
            depths=(2, 2, 4, 2),
            num_heads=(4, 4, 8, 8),
            mlp_ratio=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            pat_merg_rf=4,
            out_indices=(0, 1, 2, 3),
    ):
        super(TransMatch, self).__init__()

        # Optional Convolution
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.opt_conv = decoder.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        # self.c2 = decoder.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        # LWSA
        backbone = lwsa.LWSA(patch_size=patch_size,
                             in_chans=in_chans,
                             embed_dim=embed_dim,
                             depths=depths,
                             num_heads=num_heads,
                             window_size=window_size,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias,
                             drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate,
                             patch_norm=patch_norm,
                             use_checkpoint=use_checkpoint,
                             out_indices=out_indices,
                             pat_merg_rf=pat_merg_rf)
        self.moving_backbone = backbone
        self.fixed_backbone = backbone

        # LWCA
        lwca_config = dict(patch_size=patch_size,
                           embed_dim=embed_dim,
                           depths=depths,
                           num_heads=num_heads,
                           window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           pat_merg_rf=pat_merg_rf,
                           qkv_bias=qkv_bias,
                           drop_rate=drop_rate,
                           drop_path_rate=drop_path_rate,
                           patch_norm=patch_norm,
                           use_checkpoint=use_checkpoint,
                           out_indices=out_indices)
        self.crossattn1 = lwca.LWCA(dim_diy=96, **lwca_config)
        self.crossattn2 = lwca.LWCA(dim_diy=192, **lwca_config)
        self.crossattn3 = lwca.LWCA(dim_diy=384, **lwca_config)
        self.crossattn4 = lwca.LWCA(dim_diy=768, **lwca_config)

        self.up0 = decoder.DecoderBlock(768,
                                        384,
                                        skip_channels=384,
                                        use_batchnorm=False)
        self.up1 = decoder.DecoderBlock(384,
                                        192,
                                        skip_channels=192,
                                        use_batchnorm=False)
        self.up2 = decoder.DecoderBlock(192,
                                        96,
                                        skip_channels=96,
                                        use_batchnorm=False)
        self.up3 = decoder.DecoderBlock(96,
                                        48,
                                        skip_channels=48,
                                        use_batchnorm=False)
        self.up4 = decoder.DecoderBlock(48,
                                        16,
                                        skip_channels=16,
                                        use_batchnorm=False)

        self.up = nn.Upsample(scale_factor=2,
                              mode='trilinear',
                              align_corners=False)

    def forward(self, source, target, **kwargs):

        # Batch, channel, height, width, depth
        input_fusion = torch.cat((source, target), dim=1)

        x_s1 = self.avg_pool(input_fusion)
        f4 = self.opt_conv(x_s1)
        # f5 = self.c2(input_fusion)

        mov_feat_4, mov_feat_8, mov_feat_16, mov_feat_32 = self.moving_backbone(
            source)
        fix_feat_4, fix_feat_8, fix_feat_16, fix_feat_32 = self.fixed_backbone(
            target)

        # LWCA module
        mov_feat_4_cross = self.crossattn1(mov_feat_4, fix_feat_4)
        mov_feat_8_cross = self.crossattn2(mov_feat_8, fix_feat_8)
        mov_feat_16_cross = self.crossattn3(mov_feat_16, fix_feat_16)
        mov_feat_32_cross = self.crossattn4(mov_feat_32, fix_feat_32)

        fix_feat_4_cross = self.crossattn1(fix_feat_4, mov_feat_4)
        fix_feat_8_cross = self.crossattn2(fix_feat_8, mov_feat_8)
        fix_feat_16_cross = self.crossattn3(fix_feat_16, mov_feat_16)
        fix_feat_32_cross = self.crossattn4(fix_feat_32, mov_feat_32)

        # try concat mov_feat_32 and fix_feat_32
        x = self.up0(mov_feat_32_cross, mov_feat_16_cross, fix_feat_16_cross)
        x = self.up1(x, mov_feat_8_cross, fix_feat_8_cross)
        x = self.up2(x, mov_feat_4_cross, fix_feat_4_cross)
        x = self.up3(x, f4)
        x = self.up(x)

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
