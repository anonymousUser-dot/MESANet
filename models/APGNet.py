import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel
from models.CAGNet import Model as CAGModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGModel):
    """Adaptive Patch Generation Network.

    APG-Net changes the bottom of the IMTS backbone. Instead of committing to a
    single fixed patch partition, it creates short/base/long deployment-valid
    patch proposals, projects them onto the base patch grid, and uses a closed
    selector to form the patch token before the CAG multiresolution admission
    path. The value statistics used by the decoder stay on the base grid, so the
    model tests adaptive patch generation without destroying the amplitude
    coordinate.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        route_in = 12 + 5 * self.d_model
        self.patch_score = MLP(route_in, self.hidden, 3, self.dropout)
        self.patch_token_norm = nn.LayerNorm(self.d_model)
        last = self.patch_score.net[-1]
        if isinstance(last, nn.Linear):
            bias = torch.tensor([-1.0, 1.5, -1.0], dtype=last.bias.dtype)
            if "open_patch" in self.mode:
                bias = torch.zeros(3, dtype=last.bias.dtype)
            elif "very_closed_patch" in self.mode:
                bias = torch.tensor([-2.0, 3.0, -2.0], dtype=last.bias.dtype)
            with torch.no_grad():
                last.bias.copy_(bias)

    @staticmethod
    def _resize_grid(x: Tensor, target: int) -> Tensor:
        if x.size(2) == target:
            return x
        if x.dim() == 3:
            y = x.unsqueeze(-1)
            return Model._resize_grid(y, target).squeeze(-1)
        batch, n_vars, n_patch, dim = x.shape
        flat = x.permute(0, 1, 3, 2).reshape(batch * n_vars * dim, 1, n_patch)
        if n_patch > target:
            out = F.adaptive_avg_pool1d(flat, target)
        else:
            out = F.interpolate(flat, size=target, mode="nearest")
        return out.view(batch, n_vars, dim, target).permute(0, 1, 3, 2)

    def _proposal(self, x: Tensor, x_mask: Tensor, t: Tensor, patch_len: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        old_patch_len = self.patch_len
        self.patch_len = max(2, int(patch_len))
        try:
            stat, center_t, has_obs = AMPGModel._patch_stats(self, x, x_mask, t)
            token, _, _, alias_gate, _ = AMPGModel._make_tokens(self, stat, center_t)
        finally:
            self.patch_len = old_patch_len
        return token, stat, has_obs, alias_gate

    def _fixed_branch(self, x: Tensor, x_mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        base_token, base_stat, base_obs, base_alias = self._proposal(x, x_mask, t, self.patch_len)
        target = base_token.size(2)
        short_len = max(2, self.patch_len // 2)
        long_len = max(2, self.patch_len * 2)
        short_token, _, short_obs, short_alias = self._proposal(x, x_mask, t, short_len)
        long_token, _, long_obs, long_alias = self._proposal(x, x_mask, t, long_len)
        short_token = self._resize_grid(short_token, target)
        long_token = self._resize_grid(long_token, target)
        short_obs = self._resize_grid(short_obs, target)
        long_obs = self._resize_grid(long_obs, target)
        short_alias = self._resize_grid(short_alias, target)
        long_alias = self._resize_grid(long_alias, target)

        if "short_patch_only" in self.mode:
            token = short_token
            alias = short_alias
            obs = torch.maximum(base_obs, short_obs)
        elif "long_patch_only" in self.mode:
            token = long_token
            alias = long_alias
            obs = torch.maximum(base_obs, long_obs)
        elif "mean_patch" in self.mode:
            token = (short_token + base_token + long_token) / 3.0
            alias = (short_alias + base_alias + long_alias) / 3.0
            obs = torch.maximum(base_obs, torch.maximum(short_obs, long_obs))
        else:
            proposals = torch.stack([short_token, base_token, long_token], dim=-2)
            route = torch.cat(
                [
                    base_stat,
                    base_token,
                    short_token,
                    long_token,
                    (short_token - base_token).abs(),
                    (long_token - base_token).abs(),
                ],
                dim=-1,
            )
            logits = self.patch_score(route)
            if "no_short_patch" in self.mode:
                logits[..., 0] = -1.0e4
            if "no_long_patch" in self.mode:
                logits[..., 2] = -1.0e4
            alpha = torch.softmax(logits, dim=-1).unsqueeze(-1)
            token = (alpha * proposals).sum(dim=-2)
            alias = (
                alpha[..., 0:1, :] * short_alias.unsqueeze(-2)
                + alpha[..., 1:2, :] * base_alias.unsqueeze(-2)
                + alpha[..., 2:3, :] * long_alias.unsqueeze(-2)
            ).sum(dim=-2)
            obs = torch.maximum(base_obs, torch.maximum(short_obs, long_obs))
        return self.patch_token_norm(token), base_stat, obs, alias
