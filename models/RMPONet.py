import math

import torch
from torch import Tensor

from models.AMPGNet import Model as AMPGModel
from models.CAGNet import Model as CAGModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGModel):
    """Robust Mechanism Patch Operator Network.

    RMPO-Net is a native forecasting backbone that changes the observation-to-
    patch coordinate before graph mixing. Wearable streams often contain short
    sensor shocks, contact artifacts, and missingness bursts. A plain patch
    mean/variance can turn those local corruptions into the token itself. RMPO
    first estimates a local trend inside each mechanism-visible patch, clips
    residuals around that trend with a differentiable Huber/Winsorized rule,
    and then forms the admitted CAGNet patch token from robust statistics.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)

    def _clip_factor(self, density: Tensor) -> Tensor:
        base = 1.5
        if "tight_robust" in self.mode:
            base = 1.0
        elif "loose_robust" in self.mode:
            base = 2.5
        elif "very_loose_robust" in self.mode:
            base = 4.0
        if "no_density_robust" in self.mode:
            return torch.full_like(density, base)
        return base * (0.75 + 0.50 * density.clamp(0.0, 1.0))

    def _trend_clip(self, values: Tensor, mask: Tensor, times: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        mean = (values * mask).sum(dim=-1, keepdim=True) / denom
        t_mean = (times * mask).sum(dim=-1, keepdim=True) / denom
        t_var = (mask * (times - t_mean).square()).sum(dim=-1, keepdim=True) / denom
        cov = (mask * (values - mean) * (times - t_mean)).sum(dim=-1, keepdim=True) / denom
        slope = cov / t_var.clamp_min(1.0e-4)
        fit = mean + slope * (times - t_mean)
        resid = (values - fit) * mask
        scale = torch.sqrt((resid.square()).sum(dim=-1, keepdim=True) / denom + 1.0e-4)
        density = (mask.sum(dim=-1, keepdim=True) / values.size(-1)).clamp(0.0, 1.0)
        clip = self._clip_factor(density).clamp_min(0.25)
        robust = fit + scale * clip * torch.tanh(resid / (scale * clip).clamp_min(1.0e-4))
        return robust * mask, t_mean, t_var, slope

    def _robust_fixed_patch_stats(self, x: Tensor, mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, n_vars = x.shape
        x = self._pad(x, 0.0)
        mask = self._pad(mask, 0.0)
        t = self._pad(t, 1.0)
        n_patch = x.size(1) // self.patch_len
        x_p = x.view(batch, n_patch, self.patch_len, n_vars).permute(0, 3, 1, 2)
        mask_p = mask.view(batch, n_patch, self.patch_len, n_vars).permute(0, 3, 1, 2)
        t_p = t.view(batch, n_patch, self.patch_len, 1).permute(0, 3, 1, 2).expand(-1, n_vars, -1, -1)

        robust_x, _, _, _ = self._trend_clip(x_p, mask_p, t_p)
        stat_x = robust_x if "robust_endpoints" in self.mode else x_p
        denom = mask_p.sum(dim=-1).clamp_min(1.0)
        density = mask_p.mean(dim=-1)
        mean = robust_x.sum(dim=-1) / denom
        second = (robust_x.square() * mask_p).sum(dim=-1) / denom
        var = (second - mean.square()).clamp_min(0.0)
        first_idx = mask_p.argmax(dim=-1)
        rev_idx = torch.flip(mask_p, dims=[-1]).argmax(dim=-1)
        last_idx = self.patch_len - 1 - rev_idx
        has_obs = (mask_p.sum(dim=-1) > 0).to(x.dtype)
        gather_first = first_idx.unsqueeze(-1)
        gather_last = last_idx.unsqueeze(-1)
        first_x = torch.gather(stat_x, -1, gather_first).squeeze(-1) * has_obs
        last_x = torch.gather(stat_x, -1, gather_last).squeeze(-1) * has_obs
        first_t = torch.gather(t_p, -1, gather_first).squeeze(-1) * has_obs
        last_t = torch.gather(t_p, -1, gather_last).squeeze(-1) * has_obs
        patch_start = t_p[..., 0]
        patch_end = t_p[..., -1]
        span = (last_t - first_t).clamp_min(0.0)
        recency = (patch_end - last_t).clamp_min(0.0)
        slope = (last_x - first_x) / span.clamp_min(1.0e-3)
        center_t = 0.5 * (patch_start + patch_end)
        stat = torch.stack(
            [density, has_obs, mean, var, first_x, last_x, slope, first_t, last_t, span, recency, center_t],
            dim=-1,
        )
        stat = torch.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)
        return stat, center_t.unsqueeze(-1), has_obs

    def _robust_ctpg_patch_stats(self, x: Tensor, mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, n_vars = x.shape
        n_patch = max(1, math.ceil(seq_len / self.patch_len))
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
        widths = (base_width * self.ct_log_width.exp().to(device=device, dtype=dtype)).view(1, 1, self.n_scales, 1, 1)
        if "symmetric_kernel" in self.mode:
            kernel = torch.exp(-dist.abs() / widths.clamp_min(1.0e-4))
        else:
            z = dist.clamp_min(0.0) / widths.clamp_min(1.0e-4)
            kernel = torch.exp(-0.5 * z.square().clamp_max(64.0)) * causal

        weight = kernel * m_bdt
        denom = weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        density = weight.sum(dim=-1) / kernel.sum(dim=-1).clamp_min(1.0e-6)
        has_obs = (weight.sum(dim=-1) > 1.0e-6).to(dtype)
        mean0 = (weight * x_bdt).sum(dim=-1, keepdim=True) / denom
        t_mean = (weight * t_hist).sum(dim=-1, keepdim=True) / denom
        t_var = (weight * (t_hist - t_mean).square()).sum(dim=-1, keepdim=True) / denom
        cov0 = (weight * (x_bdt - mean0) * (t_hist - t_mean)).sum(dim=-1, keepdim=True) / denom
        slope0 = cov0 / t_var.clamp_min(1.0e-4)
        fit = mean0 + slope0 * (t_hist - t_mean)
        resid = (x_bdt - fit) * m_bdt
        scale = torch.sqrt((weight * resid.square()).sum(dim=-1, keepdim=True) / denom + 1.0e-4)
        clip = self._clip_factor(density.unsqueeze(-1)).clamp_min(0.25)
        robust_x = fit + scale * clip * torch.tanh(resid / (scale * clip).clamp_min(1.0e-4))

        mean = (weight * robust_x).sum(dim=-1) / denom.squeeze(-1)
        second = (weight * robust_x.square()).sum(dim=-1) / denom.squeeze(-1)
        var = (second - mean.square()).clamp_min(0.0)
        cov = (weight * (robust_x - mean.unsqueeze(-1)) * (t_hist - t_mean)).sum(dim=-1) / denom.squeeze(-1)
        slope = cov / t_var.squeeze(-1).clamp_min(1.0e-4)
        span = (2.0 * torch.sqrt(t_var.squeeze(-1))).clamp(0.0, 1.0)
        first_t = (t_mean.squeeze(-1) - 0.5 * span).clamp(0.0, 1.0)
        last_t = (t_mean.squeeze(-1) + 0.5 * span).clamp(0.0, 1.0)
        first_x = mean - 0.5 * slope * span
        last_x = mean + 0.5 * slope * span
        center_scalar = centers.view(1, 1, 1, n_patch)
        recency = (center_scalar - t_mean.squeeze(-1)).clamp_min(0.0)
        center = center_scalar.expand(batch, n_vars, self.n_scales, n_patch)
        scale_stat = torch.stack(
            [density, has_obs, mean, var, first_x, last_x, slope, first_t, last_t, span, recency, center],
            dim=-1,
        )
        scale_stat = torch.nan_to_num(scale_stat, nan=0.0, posinf=0.0, neginf=0.0)
        self._ct_scale_stat = scale_stat
        mid = min(1, self.n_scales - 1)
        stat = scale_stat[:, :, mid, :, :]
        patch_has_obs = has_obs.max(dim=2).values
        return stat, center_t, patch_has_obs

    def _fixed_branch(self, x: Tensor, x_mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        stat, center_t, has_obs = self._robust_fixed_patch_stats(x, x_mask, t)
        token, _, _, alias_gate, _ = AMPGModel._make_tokens(self, stat, center_t)
        return token, stat, has_obs, alias_gate

    def _scale_branch(self, x: Tensor, x_mask: Tensor, t: Tensor, fixed_stat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stat, center_t, has_obs = self._robust_ctpg_patch_stats(x, x_mask, t)
        if self._ct_scale_stat is None:
            raise RuntimeError("RMPO-Net robust CTPG stats missing.")
        scale_stat = self._ct_scale_stat
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
            bias = self.state_scale_bias(torch.cat([state_stat, state], dim=-1)).permute(0, 1, 3, 2)
            logits = logits + bias
        alpha = torch.softmax(logits, dim=2).unsqueeze(-1)
        token = (alpha * token_k).sum(dim=2)
        fused_stat = (alpha * scale_stat).sum(dim=2)
        token = self.ct_norm(token + self.var_emb.to(token.device, token.dtype).view(1, n_vars, 1, self.d_model))
        return token, fused_stat, has_obs
