# Copyright (c) MMIPT. All rights reserved.
import torch
import torch.nn.functional as F
from torch import Tensor, fft, nn


class FourierTransform(nn.Module):

    def __init__(self, img_size):
        super(FourierTransform, self).__init__()
        self.img_size = img_size

    def forward(self, feats: Tensor, *args, **kwargs):
        ndim = feats.ndim - 2
        if ndim == 2:
            p1 = (self.img_size[-1] - feats.shape[-1]) // 2
            p2 = (self.img_size[-2] - feats.shape[-2]) // 2
            pad = (p1, p1, p2, p2)
        elif ndim == 3:
            p1 = (self.img_size[-1] - feats.shape[-1]) // 2
            p2 = (self.img_size[-2] - feats.shape[-2]) // 2
            p3 = (self.img_size[-3] - feats.shape[-3]) // 2
            pad = (p1, p1, p2, p2, p3, p3)

        disps = []
        for i in range(ndim):
            # DFT
            ifft = fft.fftshift(fft.fftn(feats[0, i]))
            # Padding
            ifft = F.pad(ifft, pad, "constant", 0)
            ## iDFT
            disps.append(torch.real(fft.ifftn(fft.ifftshift(ifft))))
        flow = torch.stack(disps).unsqueeze_(0)
        return flow
