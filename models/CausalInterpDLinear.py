"""History-only interpolation followed by the public DLinear forecaster."""

from __future__ import annotations

import torch
from torch import Tensor

from models.DLinear import Model as DLinearModel


class Model(DLinearModel):
    """A regular-grid sanity baseline with no access to future targets.

    Missing entries are linearly interpolated from observations inside the
    history window. Values outside the first/last observation are carried from
    the nearest observed endpoint. The interpolated history is then passed to
    the unchanged DLinear architecture.
    """

    @staticmethod
    def _history_interpolate(x: Tensor, t: Tensor | None, mask: Tensor | None) -> Tensor:
        if mask is None:
            return x
        mask = mask > 0
        batch, length, n_vars = x.shape
        if t is None:
            grid = torch.arange(length, device=x.device, dtype=x.dtype)
            times = grid.view(1, -1).expand(batch, -1)
        else:
            times = t[..., 0] if t.ndim == 3 else t
            times = times.to(device=x.device, dtype=x.dtype)
        index = torch.arange(length, device=x.device).view(1, length, 1).expand(batch, -1, n_vars)
        left = torch.where(mask, index, -torch.ones_like(index)).cummax(dim=1).values
        right_seed = torch.where(mask, index, torch.full_like(index, length))
        right = torch.flip(torch.flip(right_seed, dims=[1]).cummin(dim=1).values, dims=[1])
        has_left, has_right = left >= 0, right < length
        left_idx = left.clamp(0, length - 1)
        right_idx = right.clamp(0, length - 1)
        x_left, x_right = x.gather(1, left_idx), x.gather(1, right_idx)
        time_grid = times.unsqueeze(-1).expand(-1, -1, n_vars)
        t_left = time_grid.gather(1, left_idx)
        t_right = time_grid.gather(1, right_idx)
        weight = ((time_grid - t_left) / (t_right - t_left).clamp_min(1e-8)).clamp(0.0, 1.0)
        interpolated = x_left + weight * (x_right - x_left)
        interpolated = torch.where(has_left & ~has_right, x_left, interpolated)
        interpolated = torch.where(~has_left & has_right, x_right, interpolated)
        interpolated = torch.where(has_left | has_right, interpolated, torch.zeros_like(interpolated))
        return torch.where(mask, x, interpolated)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor | None = None,
        x_mask: Tensor | None = None,
        **kwargs,
    ):
        filled = self._history_interpolate(x, x_mark, x_mask)
        return super().forward(x=filled, **kwargs)
