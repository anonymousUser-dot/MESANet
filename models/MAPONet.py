import math

import torch
import torch.nn as nn
from torch import Tensor

from models.AMPGNet import MLP
from models.CAGNet import Model as CAGModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGModel):
    """Mechanism-Adaptive Patch Operator Network.

    MAPONet keeps CAGNet's level-preserving admission backbone but moves the
    main innovation into patch construction.  For each variable and patch, a
    deployment-visible mechanism state controls the causal continuous-time
    kernel width before the patch token is formed.  Dense, high-activity
    regions keep local kernels; sparse and smooth regions can integrate over a
    wider support.  The later CAGNet admission then decides whether this
    mechanism-adaptive patch should replace the fixed level patch.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.mapo_width_shift = MLP(12, self.hidden, self.n_scales, self.dropout)
        self.mapo_score_bias = MLP(24, self.hidden, 1, self.dropout)
        self.mapo_stat_norm = nn.LayerNorm(12)
        for head in (self.mapo_width_shift, self.mapo_score_bias):
            last = head.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _mechanism_width_factor(self, fixed_stat: Tensor) -> Tensor:
        density = fixed_stat[..., 0].clamp(0.05, 1.0)
        activity = fixed_stat[..., 3].clamp_min(0.0)
        trend = fixed_stat[..., 6].abs()
        recency = fixed_stat[..., 10].clamp(0.0, 1.0)

        sparse_stretch = density.rsqrt().clamp(1.0, 4.0)
        if "no_density_kernel" in self.mode or "fixed_kernel" in self.mode:
            sparse_stretch = torch.ones_like(sparse_stretch)

        if "no_variance_guard" in self.mode or "fixed_kernel" in self.mode:
            smooth_gate = torch.ones_like(activity)
        else:
            smooth_gate = (1.0 / (1.0 + 4.0 * activity + 0.05 * trend)).clamp(0.0, 1.0)

        stretch = 1.0 + (sparse_stretch - 1.0) * smooth_gate
        if "no_recency_kernel" not in self.mode and "fixed_kernel" not in self.mode:
            stretch = stretch * (1.0 + 0.25 * recency * (1.0 - density))
        if "conservative_kernel" in self.mode:
            stretch = 1.0 + 0.5 * (stretch - 1.0)
        elif "open_kernel" in self.mode:
            stretch = 1.0 + 1.5 * (stretch - 1.0)
        return stretch.clamp(0.5, 5.0)

    def _adaptive_scale_stats(
        self,
        x: Tensor,
        mask: Tensor,
        t: Tensor,
        fixed_stat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, n_vars = x.shape
        n_patch = fixed_stat.size(2)
        dtype = x.dtype
        device = x.device

        centers = (torch.arange(n_patch, device=device, dtype=dtype) + 1.0) / max(n_patch, 1)
        center_t = centers.view(1, 1, n_patch, 1).expand(batch, n_vars, -1, -1)
        x_bdt = x.permute(0, 2, 1).unsqueeze(2).unsqueeze(2)
        m_bdt = mask.permute(0, 2, 1).unsqueeze(2).unsqueeze(2)
        t_hist = t[:, :, 0].view(batch, 1, 1, 1, seq_len)
        c = centers.view(1, 1, 1, n_patch, 1)

        base_width = 1.0 / max(n_patch, 1)
        dist = c - t_hist
        causal = (dist >= 0).to(dtype)
        width_factor = self._mechanism_width_factor(fixed_stat).unsqueeze(2).unsqueeze(-1)
        base_scale = self.ct_log_width.exp().to(device=device, dtype=dtype).view(1, 1, self.n_scales, 1, 1)
        if "fixed_kernel" in self.mode or "no_learned_width" in self.mode:
            learned = torch.ones(batch, n_vars, self.n_scales, n_patch, 1, device=device, dtype=dtype)
        else:
            learned_shift = torch.tanh(self.mapo_width_shift(fixed_stat)).permute(0, 1, 3, 2).unsqueeze(-1)
            learned = torch.exp(0.5 * learned_shift)
        widths = (base_width * base_scale * width_factor * learned).clamp_min(1.0e-4)

        if "box_kernel" in self.mode:
            kernel = (dist.abs() <= widths).to(dtype)
        elif "exp_kernel" in self.mode:
            kernel = torch.exp(-dist.clamp_min(0.0) / widths) * causal
        else:
            z = dist.clamp_min(0.0) / widths
            kernel = torch.exp(-0.5 * z.square().clamp_max(64.0)) * causal

        weight = kernel * m_bdt
        denom = weight.sum(dim=-1).clamp_min(1.0e-6)
        density = weight.sum(dim=-1) / kernel.sum(dim=-1).clamp_min(1.0e-6)
        has_obs = (weight.sum(dim=-1) > 1.0e-6).to(dtype)
        mean = (weight * x_bdt).sum(dim=-1) / denom
        second = (weight * x_bdt.square()).sum(dim=-1) / denom
        var = (second - mean.square()).clamp_min(0.0)
        t_mean = (weight * t_hist).sum(dim=-1) / denom
        t_second = (weight * t_hist.square()).sum(dim=-1) / denom
        t_var = (t_second - t_mean.square()).clamp_min(1.0e-6)
        cov = (weight * (x_bdt - mean.unsqueeze(-1)) * (t_hist - t_mean.unsqueeze(-1))).sum(dim=-1) / denom
        slope = cov / t_var.clamp_min(1.0e-4)
        span = (2.0 * torch.sqrt(t_var)).clamp(0.0, 1.0)
        first_t = (t_mean - 0.5 * span).clamp(0.0, 1.0)
        last_t = (t_mean + 0.5 * span).clamp(0.0, 1.0)
        first_x = mean - 0.5 * slope * span
        last_x = mean + 0.5 * slope * span
        center_scalar = centers.view(1, 1, 1, n_patch)
        recency = (center_scalar - t_mean).clamp_min(0.0)
        center = center_scalar.expand(batch, n_vars, self.n_scales, n_patch)
        scale_stat = torch.stack(
            [density, has_obs, mean, var, first_x, last_x, slope, first_t, last_t, span, recency, center],
            dim=-1,
        )
        scale_stat = torch.nan_to_num(scale_stat, nan=0.0, posinf=0.0, neginf=0.0)
        return scale_stat, center_t, has_obs.max(dim=2).values

    def _scale_branch(self, x: Tensor, x_mask: Tensor, t: Tensor, fixed_stat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scale_stat, center_t, has_obs = self._adaptive_scale_stats(x, x_mask, t, fixed_stat)
        self._ct_scale_stat = scale_stat
        batch, n_vars, n_scales, n_patch, _ = scale_stat.shape
        fixed_stat = fixed_stat[:, :, :n_patch, :]
        var = self.var_emb.to(scale_stat.device, scale_stat.dtype).view(1, n_vars, 1, 1, self.d_model)
        var = var.expand(batch, -1, n_scales, n_patch, -1)
        te = self.time_enc(center_t).unsqueeze(2).expand(-1, -1, n_scales, -1, -1)
        base = torch.cat([scale_stat, te, var], dim=-1)
        token_k = self.ct_scale_op(base)
        logits = self.ct_scale_score(base).squeeze(-1)

        if "no_state_scale" not in self.mode:
            state_stat = self._value_only_stat(fixed_stat) if "value_only_state" in self.mode else fixed_stat
            state = self.state_emb(state_stat)
            logits = logits + self.state_scale_bias(torch.cat([state_stat, state], dim=-1)).permute(0, 1, 3, 2)
        if "no_mapo_score" not in self.mode:
            fixed_expand = fixed_stat.unsqueeze(2).expand_as(scale_stat)
            score_bias = self.mapo_score_bias(torch.cat([fixed_expand, scale_stat], dim=-1)).squeeze(-1)
            logits = logits + score_bias

        if "single_scale" in self.mode:
            idx = min(1, n_scales - 1)
            alpha = torch.zeros(batch, n_vars, n_scales, n_patch, 1, device=scale_stat.device, dtype=scale_stat.dtype)
            alpha[:, :, idx, :, :] = 1.0
        elif "uniform_scale" in self.mode:
            alpha = torch.full((batch, n_vars, n_scales, n_patch, 1), 1.0 / n_scales, device=scale_stat.device, dtype=scale_stat.dtype)
        else:
            alpha = torch.softmax(logits, dim=2).unsqueeze(-1)
        token = (alpha * token_k).sum(dim=2)
        fused_stat = (alpha * scale_stat).sum(dim=2)
        token = self.ct_norm(token + self.var_emb.to(token.device, token.dtype).view(1, n_vars, 1, self.d_model))
        self.latest_mapo_width_factor = self._mechanism_width_factor(fixed_stat).detach()
        return token, fused_stat, has_obs
