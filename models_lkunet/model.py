import torch
from torch import nn
from .model_util import Warp


# @MODELS.register_module()
class LKUNet(nn.Module):

    def __init__(
        self,
        img_szie,
        in_dim=2,
        out_dim=3,
        embed_dim=16,
        kernel_size=5,
        bias_opt=True,
    ):
        super(LKUNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim

        self.eninput = self.encoder(
            self.in_dim,
            self.embed_dim,
            bias=bias_opt,
        )
        self.ec1 = self.encoder(
            self.embed_dim,
            self.embed_dim,
            bias=bias_opt,
        )
        self.ec2 = self.encoder(
            self.embed_dim,
            self.embed_dim * 2,
            stride=2,
            bias=bias_opt,
        )
        self.ec3 = LK_encoder(
            self.out_dim,
            self.embed_dim * 2,
            self.embed_dim * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias_opt,
        )
        self.ec4 = self.encoder(
            self.embed_dim * 2,
            self.embed_dim * 4,
            stride=2,
            bias=bias_opt,
        )
        self.ec5 = LK_encoder(
            self.out_dim,
            self.embed_dim * 4,
            self.embed_dim * 4,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias_opt,
        )
        self.ec6 = self.encoder(
            self.embed_dim * 4,
            self.embed_dim * 8,
            stride=2,
            bias=bias_opt,
        )
        self.ec7 = LK_encoder(
            self.out_dim,
            self.embed_dim * 8,
            self.embed_dim * 8,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias_opt,
        )
        self.ec8 = self.encoder(
            self.embed_dim * 8,
            self.embed_dim * 8,
            stride=2,
            bias=bias_opt,
        )
        self.ec9 = LK_encoder(
            self.out_dim,
            self.embed_dim * 8,
            self.embed_dim * 8,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias_opt,
        )

        self.up1 = self.decoder(self.embed_dim * 8, self.embed_dim * 8)
        self.dc1 = self.encoder(
            self.embed_dim * 8 + self.embed_dim * 8,
            self.embed_dim * 8,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.dc2 = self.encoder(
            self.embed_dim * 8,
            self.embed_dim * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        self.up2 = self.decoder(self.embed_dim * 4, self.embed_dim * 4)
        self.dc3 = self.encoder(
            self.embed_dim * 4 + self.embed_dim * 4,
            self.embed_dim * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.dc4 = self.encoder(
            self.embed_dim * 4,
            self.embed_dim * 2,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        self.up3 = self.decoder(self.embed_dim * 2, self.embed_dim * 2)
        self.dc5 = self.encoder(
            self.embed_dim * 2 + self.embed_dim * 2,
            self.embed_dim * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.dc6 = self.encoder(
            self.embed_dim * 4,
            self.embed_dim * 2,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        self.up4 = self.decoder(self.embed_dim * 2, self.embed_dim * 2)
        self.dc7 = self.encoder(
            self.embed_dim * 2 + self.embed_dim * 1,
            self.embed_dim * 2,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.dc8 = self.encoder(
            self.embed_dim * 2,
            self.embed_dim * 2,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        self.dc9 = self.outputs(
            self.embed_dim * 2,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.warp = Warp(img_size=img_szie, normalization=True)

    def encoder(self,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                batchnorm=False):
        conv_fn = getattr(nn, "Conv%dd" % self.out_dim)
        bn_fn = getattr(nn, "BatchNorm%dd" % self.out_dim)
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias),
                bn_fn(out_channels),
                nn.PReLU(),
            )
        else:
            layer = nn.Sequential(
                conv_fn(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias),
                nn.PReLU(),
            )
        return layer

    def decoder(self,
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=True):
        deconv_fn = getattr(nn, "ConvTranspose%dd" % self.out_dim)
        layer = nn.Sequential(
            deconv_fn(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias),
            nn.PReLU(),
        )
        return layer

    def outputs(self,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                batchnorm=False):
        conv_fn = getattr(nn, "Conv%dd" % self.out_dim)
        bn_fn = getattr(nn, "BatchNorm%dd" % self.out_dim)
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias),
                bn_fn(out_channels),
                nn.Tanh(),
            )
        else:
            layer = nn.Sequential(
                conv_fn(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias),
                nn.Softsign(),
            )
        return layer

    def forward(self, source, target, **kwargs):
        x_in = torch.cat((source, target), 1)

        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)
        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)
        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        flow = self.dc9(d3)
        def_out = self.warp(source, flow=flow)
        return def_out, flow


# @MODELS.register_module()
class CascadeLKUNet(nn.Module):

    def __init__(
        self,
        img_size,
        in_dim=2,
        out_dim=3,
        embed_dim=16,
        kernel_size=5,
        bias_opt=True,
    ):
        super(CascadeLKUNet, self).__init__()
        from .model_util import Warp
        self.net1 = LKUNet(
            in_dim=in_dim,
            out_dim=out_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            bias_opt=bias_opt)
        self.warp = Warp(img_size, normalization=True)

    def forward(self, source, target, **kwargs):
        fxy_1 = self.net1(source, target)
        # x2 = self.warp(fxy_1, source)
        x2 = self.warp(source, fxy_1)

        fxy_2 = self.net1(x2, target)
        # fxy_2_ = self.warp(fxy_2, fxy_1)
        fxy_2_ = self.warp(fxy_1, fxy_2)

        fxy_2_ = fxy_2_ + fxy_2
        # x3 = self.warp(fxy_2_, source)
        x3 = self.warp(source, fxy_2_)

        fxy_3 = self.net1(x3, target)
        # fxy_3_ = self.warp(fxy_3, fxy_2_)
        fxy_3_ = self.warp(fxy_2_, fxy_3)
        fxy_3_ = fxy_3_ + fxy_3

        return fxy_3_


class LK_encoder(nn.Module):

    def __init__(self,
                 out_dim,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 bias=False,
                 batchnorm=False):
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.batchnorm = batchnorm

        super(LK_encoder, self).__init__()

        self.layer_regularKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias,
            batchnorm=self.batchnorm)
        self.layer_largeKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            batchnorm=self.batchnorm)
        self.layer_oneKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
            batchnorm=self.batchnorm)
        self.layer_nonlinearity = nn.PReLU()

    def encoder_LK_encoder(self,
                           in_channels,
                           out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False,
                           batchnorm=False):
        conv_fn = getattr(nn, "Conv%dd" % self.out_dim)
        bn_fn = getattr(nn, "BatchNorm%dd" % self.out_dim)
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias),
                bn_fn(out_channels),
            )
        else:
            layer = nn.Sequential(
                conv_fn(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias))
        return layer

    def forward(self, inputs):
        # print(self.layer_regularKernel)
        regularKernel = self.layer_regularKernel(inputs)
        largeKernel = self.layer_largeKernel(inputs)
        oneKernel = self.layer_oneKernel(inputs)
        # if self.layer_indentity:
        outputs = regularKernel + largeKernel + oneKernel + inputs
        # else:
        # outputs = regularKernel + largeKernel + oneKernel
        # if self.batchnorm:
        # outputs = self.layer_batchnorm(self.layer_batchnorm)
        return self.layer_nonlinearity(outputs)
