# Copyright (c) MMIPT. All rights reserved.
import torch
from torch import nn


class Conv3dReLU(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super().__init__(conv, nm, relu)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels + skip_channels,
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
        self.conv3 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None, skip2=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        if skip2 is not None:
            x = torch.cat([x, skip2], dim=1)
            x = self.conv1(x)
        if skip2 is None:
            x = self.conv3(x)
        x = self.conv2(x)
        return x
