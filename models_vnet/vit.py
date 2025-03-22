import math
from typing import Optional

import torch
from torch import Tensor, nn
from .model_util import Warp, DefaultFlow


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
        conv = nn.Conv3d(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)  # differ from transmorph
        bn = nn.BatchNorm3d(out_channels)
        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(in_channels + skip_channels,
                                out_channels,
                                kernel_size=3,
                                padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv3dReLU(out_channels,
                                out_channels,
                                kernel_size=3,
                                padding=1,
                                use_batchnorm=use_batchnorm)
        self.up = nn.Upsample(scale_factor=2,
                              mode='trilinear',
                              align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class CNNEncoder(nn.Module):

    def __init__(self, encoder_channels, down_num):
        super().__init__()
        self.down_num = down_num
        in_channels, channels = encoder_channels[0], encoder_channels[1:]
        self.inc = DoubleConv(in_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        feats = self.down2(x2)

        feats_down = feats
        features = [x1, x2, feats]
        for _ in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)
            features.append(feats_down)
        return feats, features[::-1]


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 hidden_size,
                 drop_rate,
                 encoder_channels=(2, 16, 32, 32),
                 down_num=2,
                 down_factor=2):
        super(Embeddings, self).__init__()
        n_patches = int((img_size[0] / 2**down_factor // patch_size[0]) *
                        (img_size[1] / 2**down_factor // patch_size[1]) *
                        (img_size[2] / 2**down_factor // patch_size[2]))

        self.hybrid_model = CNNEncoder(encoder_channels, down_num)
        self.patch_embeddings = nn.Conv3d(in_channels=encoder_channels[-1],
                                          out_channels=hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: Tensor):
        x, features = self.hybrid_model(x)
        # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Attention(nn.Module):

    def __init__(self, num_heads, hidden_size, attn_drop_rate, vis=False):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attn_heads = num_heads
        self.attn_head_size = int(hidden_size / self.num_attn_heads)
        self.all_head_size = self.num_attn_heads * self.attn_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_drop_rate)
        self.proj_dropout = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x: Tensor):
        new_x_shape = (x.shape[:-1] +
                       (self.num_attn_heads, self.attn_head_size))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: Tensor):
        query_layer = self.transpose_for_scores(self.query(x))
        key_layer = self.transpose_for_scores(self.key(x))
        value_layer = self.transpose_for_scores(self.value(x))

        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.attn_head_size)
        attn_probs = self.softmax(attn_scores)
        weights = attn_probs if self.vis else None
        attn_probs = self.attn_dropout(attn_probs)

        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        attn_output = self.out(context_layer)
        attn_output = self.proj_dropout(attn_output)
        return attn_output, weights


class Mlp(nn.Module):

    def __init__(self, hidden_size, mlp_dim, drop_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, hidden_size, mlp_dim, num_heads, attn_drop_rate,
                 drop_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, drop_rate)
        self.attn = Attention(num_heads, hidden_size, attn_drop_rate)

    def forward(self, x: Tensor):
        # attn
        res = x
        x = self.attention_norm(x)
        x, _ = self.attn(x)
        x = x + res
        # mlp
        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + res
        return x


class Encoder(nn.Module):

    def __init__(self, hidden_size, mlp_dim, num_heads, attn_drop_rate,
                 drop_rate, num_layers):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList([
            Block(hidden_size, mlp_dim, num_heads, attn_drop_rate, drop_rate)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x: Tensor):
        for layer_block in self.layer:
            x = layer_block(x)
        x = self.encoder_norm(x)
        return x


class Transformer(nn.Module):

    def __init__(self, img_size, patch_size, hidden_size, mlp_dim, num_heads,
                 attn_drop_rate, drop_rate, num_layers):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size,
                                     patch_size,
                                     hidden_size,
                                     drop_rate,
                                     encoder_channels=[2, 16, 32, 32],
                                     down_num=2,
                                     down_factor=2)
        self.encoder = Encoder(hidden_size, mlp_dim, num_heads, attn_drop_rate,
                               drop_rate, num_layers)

    def forward(self, x):
        x, feats = self.embeddings(x)
        x = self.encoder(x)
        return x, feats


class DecoderCup(nn.Module):

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        head_channels,
        decoder_channels,
        skip_channels,
        down_factor,
        n_skip,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.down_factor = down_factor
        self.n_skip = n_skip

        self.conv_more = Conv3dReLU(hidden_size,
                                    head_channels,
                                    kernel_size=3,
                                    padding=1,
                                    use_batchnorm=True)

        in_channels = [head_channels] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(
                in_channels, decoder_channels, skip_channels)
        ])

    def forward(self, hidden_states: Tensor, features=None):
        # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        B, n_patch, hidden = hidden_states.shape
        l, h, w = (
            (self.img_size[0] // 2**self.down_factor // self.patch_size[0]),
            (self.img_size[1] // 2**self.down_factor // self.patch_size[0]),
            (self.img_size[2] // 2**self.down_factor // self.patch_size[0]),
        )
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = None
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            x = decoder_block(x, skip=skip)
        return x


class ViTVNet(nn.Module):

    def __init__(self,
                 img_size,
                 patch_size=[8, 8, 8],
                 hidden_size=252,
                 mlp_dim=3072,
                 num_heads=12,
                 attn_drop_rate=0.0,
                 drop_rate=0.1,
                 num_layers=12,
                 head_channels=512,
                 decoder_channels=[96, 48, 32, 32, 16],
                 skip_channels=[32, 32, 32, 32, 16],
                 down_factor=2,
                 n_skip=5):
        super().__init__()
        self.transformer = Transformer(img_size, patch_size, hidden_size,
                                       mlp_dim, num_heads, attn_drop_rate,
                                       drop_rate, num_layers)
        self.decoder = DecoderCup(
            img_size,
            patch_size,
            hidden_size,
            head_channels,
            decoder_channels,
            skip_channels,
            down_factor,
            n_skip,
        )
        self.flow = DefaultFlow(16)
        self.warp = Warp(img_size)

    def forward(self, source, target, **kwargs):
        x = torch.cat([source, target], dim=1)
        x, feats = self.transformer(x)
        # (B, n_patch, hidden)
        x = self.decoder(x, feats)
        flow = self.flow(x)

        warped = self.warp(source, flow=flow)
        return warped, flow
