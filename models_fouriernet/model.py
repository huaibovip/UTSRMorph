import torch
from torch import nn
from .model_util import Warp
from .fourier_neck import FourierTransform


class FourierNet(nn.Module):

    def __init__(
        self,
        img_szie,
        in_dim=2,
        out_dim=3,
        embed_dim=16,
        bias_opt=True,
    ):
        super(FourierNet, self).__init__()
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
        self.ec3 = self.encoder(
            self.embed_dim * 2,
            self.embed_dim * 2,
            bias=bias_opt,
        )
        self.ec4 = self.encoder(
            self.embed_dim * 2,
            self.embed_dim * 4,
            stride=2,
            bias=bias_opt,
        )
        self.ec5 = self.encoder(
            self.embed_dim * 4,
            self.embed_dim * 4,
            bias=bias_opt,
        )
        self.ec6 = self.encoder(
            self.embed_dim * 4,
            self.embed_dim * 8,
            stride=2,
            bias=bias_opt,
        )
        self.ec7 = self.encoder(
            self.embed_dim * 8,
            self.embed_dim * 8,
            bias=bias_opt,
        )
        self.ec8 = self.encoder(
            self.embed_dim * 8,
            self.embed_dim * 16,
            stride=2,
            bias=bias_opt,
        )
        self.ec9 = self.encoder(
            self.embed_dim * 16,
            self.embed_dim * 8,
            bias=bias_opt,
        )

        self.r_up1 = self.decoder(
            self.embed_dim * 8,
            self.embed_dim * 8,
        )
        self.r_dc1 = self.encoder(
            self.embed_dim * 8 + self.embed_dim * 8,
            self.embed_dim * 8,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.r_dc2 = self.encoder(
            self.embed_dim * 8,
            self.embed_dim * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        self.r_up2 = self.decoder(
            self.embed_dim * 4,
            self.embed_dim * 4,
        )
        self.r_dc3 = self.encoder(
            self.embed_dim * 4 + self.embed_dim * 4,
            self.embed_dim * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.r_dc4 = self.encoder(
            self.embed_dim * 4,
            self.embed_dim * 2,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        self.rr_dc9 = self.outputs(
            self.embed_dim * 2,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.flow = FourierTransform(img_size=img_szie)
        self.warp = Warp(img_size=img_szie, normalization=True)

    def encoder(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        batchnorm=False,
    ):
        conv_fn = getattr(nn, "Conv%dd" % self.out_dim)
        bn_fn = getattr(nn, "BatchNorm%dd" % self.out_dim)
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                conv_fn(in_channels,
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
                # nn.Dropout(0.1),
                conv_fn(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias),
                nn.PReLU(),
            )
        return layer

    def decoder(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        deconv_fn = getattr(nn, "ConvTranspose%dd" % self.out_dim)
        layer = nn.Sequential(
            # nn.Dropout(0.1),
            deconv_fn(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      output_padding=output_padding,
                      bias=bias),
            nn.PReLU(),
        )
        return layer

    def outputs(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        batchnorm=False,
    ):
        conv_fn = getattr(nn, "Conv%dd" % self.out_dim)
        bn_fn = getattr(nn, "BatchNorm%dd" % self.out_dim)
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels,
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
                conv_fn(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias))
            # nn.Softsign())
        return layer

    def forward(self, source, target, **kwargs):
        assert source.shape[0] == 1, 'only support batch_size=1'
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

        r_d0 = torch.cat((self.r_up1(e4), e3), 1)
        r_d0 = self.r_dc1(r_d0)
        r_d0 = self.r_dc2(r_d0)
        r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)
        r_d1 = self.r_dc3(r_d1)
        r_d1 = self.r_dc4(r_d1)
        feats = self.rr_dc9(r_d1)

        if self.out_dim == 2:
            feats = feats * 64
        flow = self.flow(feats)

        def_out = self.warp(source, flow=flow)
        return def_out, flow
