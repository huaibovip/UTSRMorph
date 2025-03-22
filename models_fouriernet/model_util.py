# Copyright (c) MMIPT. All rights reserved.
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Warp(nn.Module):
    """Warp an image with given flow / dense displacement field (DDF).

    Args:
        img_size (Sequence[int]): Size of input image.
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
        img_size: Sequence[int],
        normalization: bool = False,
        align_corners: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.ndim = len(img_size)
        self.img_size = tuple(img_size)
        self.normalization = normalization
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.default_interp_mode = 'bilinear'

        # create sampling grid
        grid = self.create_grid(img_size, normalization=normalization)

        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid, persistent=False)
        # self.grid.requires_grad_(False)

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

    def forward(self,
                image: Tensor,
                flow: Tensor,
                mode: Optional[str] = None) -> Tensor:
        """
        Warp image with flow.
        Args:
            flow (Tensor): flow field of shape [batch_size, spatial_dims, ...]
            image (Tensor): input image of shape [batch_size, channels, ...]
            interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]

        Returns:
            Tensor: Warped image.
        """

        if mode is None:
            mode = self.default_interp_mode

        # warped deformation filed
        warped_grid = self.grid.to(flow) + flow

        # normalize grid values to [-1, 1] for resampler
        if not self.normalization:
            scale = torch.tensor([s - 1 for s in self.img_size])
            scale = scale.view((1, self.ndim, 1, 1, 1)).to(warped_grid)
            warped_grid = 2 * warped_grid / scale - 1.0

        # move channels dim to last position also not sure why,
        # but the channels need to be reversed
        warped_grid = warped_grid.permute(
            (0, *[i + 2 for i in range(self.ndim)], 1))
        warped_grid = warped_grid.flip(-1)

        return F.grid_sample(
            image,
            warped_grid,
            mode=mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


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

    def forward(self,
                flow: Tensor,
                image: Tensor,
                interp_mode: Optional[str] = None) -> Tensor:
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


class CompositionTransform(nn.Module):

    def __init__(self, range_flow=1.0):
        super().__init__()
        self.range_flow = range_flow

    def forward(self, flow1, flow2, sample_grid):
        size_tensor = sample_grid.shape

        # yapf: disable
        grid = sample_grid + flow1 * self.range_flow
        grid[..., 0] = (grid[..., 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
        grid[..., 1] = (grid[..., 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
        grid[..., 2] = (grid[..., 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
        return F.grid_sample(flow2, grid, mode='bilinear') + flow1
        # yapf: enable


def affine_warp(flow: Tensor,
                image: Tensor,
                mode: str = "bilinear",
                padding_mode: str = "zeros",
                align_corners: Optional[bool] = False):

    return F.grid_sample(
        input=image,
        grid=flow,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)


@torch.no_grad()
def flow_denorm(flow: Tensor, norm: bool):
    if not norm:
        return flow

    shape = flow.shape[2:]
    new_flow = torch.zeros_like(flow)
    new_flow[:, 0] = flow[:, 0] * shape[0] / 2
    new_flow[:, 1] = flow[:, 1] * shape[1] / 2
    if len(shape) == 3:
        new_flow[:, 2] = flow[:, 2] * shape[2] / 2
    return new_flow


# TODO
def flow_warp2d(x,
                flow,
                interpolation='bilinear',
                padding_mode='zeros',
                align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    # torch.meshgrid has been modified in 1.10.0 (compatibility with previous
    # versions), and will be further modified in 1.12 (Breaking Change)
    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    shape = x.shape[2:]
    device = flow.device

    # create sampling grid
    vectors = [torch.arange(0, s, device=device, dtype=x.dtype) for s in shape]
    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grids = torch.meshgrid(vectors, indexing='ij')
    else:
        grids = torch.meshgrid(vectors)
    grid = torch.stack(grids).unsqueeze(0)
    grid.requires_grad = False

    # new locations
    grid_flow = flow + grid

    # scale grid_flow to [-1,1]
    for i in range(len(shape)):
        grid_flow[:, i, ...] = 2 * (
            grid_flow[:, i, ...] / (shape[i] - 1) - 0.5)

    if len(shape) == 2:
        grid_flow = grid_flow.permute(0, 2, 3, 1)
        grid_flow = grid_flow[..., [1, 0]]
    elif len(shape) == 3:
        grid_flow = grid_flow.permute(0, 2, 3, 4, 1)
        grid_flow = grid_flow[..., [2, 1, 0]]

    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


@torch.no_grad()
def inverse_consistency(disp1: Tensor,
                        disp2: Tensor,
                        warp_fn: Callable,
                        n_iters: int = 15):
    """
    enforce inverse consistency of forward and backward transform
    """
    disp1i = disp1.clone()
    disp2i = disp2.clone()
    for _ in range(n_iters):
        disp1i = 0.5 * (disp1i - warp_fn(disp1i, disp2i))
        disp2i = 0.5 * (disp2i - warp_fn(disp2i, disp1i))

    return disp1i, disp2i
