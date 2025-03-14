import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def pdist_squared(x: Tensor):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


def mind_ssc(img: Tensor, radius=1, dilation=2, in_dim=1):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf
    # for details on the MIND-SSC descriptor
    device = img.device

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_nbh = torch.Tensor([
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 2],
        [2, 1, 1],
        [1, 2, 1],
    ]).to(device)

    # squared distances
    dist = pdist_squared(six_nbh.t().unsqueeze(0)).squeeze(0).long()
    six_nbh = six_nbh.long()

    # define comparison mask
    x, y = torch.meshgrid(
        torch.arange(6, device=device),
        torch.arange(6, device=device),
        indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_nbh.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_nbh.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, in_dim, 3, 3, 3, device=device)
    mshift1.view(-1)[torch.arange(12, device=device) * 27 +
                     idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 +
                     idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, in_dim, 3, 3, 3, device=device)
    mshift2.view(-1)[torch.arange(12, device=device) * 27 +
                     idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 +
                     idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation).to(device)
    rpad2 = nn.ReplicationPad3d(radius).to(device)

    # compute patch-ssd
    ssd = F.avg_pool3d(
        rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) -
               F.conv3d(rpad1(img), mshift2, dilation=dilation))**2),
        kernel_size,
        stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var,
                           mind_var.mean().item() * 0.001,
                           mind_var.mean().item() * 1000)
    mind = torch.exp(-mind / mind_var)

    #permute to have same ordering as C++ code
    # index = torch.as_tensor(
    #     [6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3],
    #     dtype=torch.long,
    #     device=device,
    # )
    # mind = mind[:, index, :, :, :]

    return mind
