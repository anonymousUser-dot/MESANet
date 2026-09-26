import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from utils.ExpConfigs import ExpConfigs


class Model(AMPGModel):
    """Continuous-Time Patch Graph Network.

    CTPG-Net is a bottom-up IMTS backbone. Instead of first cutting the history
    into fixed chunks and then appending gap features, it forms patch tokens by
    applying learnable continuous-time kernels directly to the irregular
    observations. The kernel widths define adaptive temporal patches, and the
    fused patch tokens are propagated by temporal and variable graph operators.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.n_scales = max(1, int(getattr(configs, "cag_num_scales", 4)))
        log_width = torch.linspace(math.log(0.5), math.log(4.0), self.n_scales, dtype=torch.float32)
        if self.n_scales == 1:
            log_width = torch.zeros(1, dtype=torch.float32)
        self.ct_log_width = nn.Parameter(log_width)
        self.ct_scale_op = MLP(12 + max(4, int(getattr(configs, "tpatchgnn_te_dim", 10))) + self.d_model, self.hidden, self.d_model, self.dropout)
        self.ct_scale_score = MLP(12 + max(4, int(getattr(configs, "tpatchgnn_te_dim", 10))) + self.d_model, self.hidden, 1, self.dropout)
        self.ct_norm = nn.LayerNorm(self.d_model)
        self.ct_stat_norm = nn.LayerNorm(12)

        self.ct_st_conv = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
        )
        self.ct_st_gate = MLP(12 + self.d_model, self.hidden, self.d_model, self.dropout)
        self.ct_st_norm = nn.LayerNorm(self.d_model)

        te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
        dec_in = self.d_model * 2 + te_dim + 8
        self.ct_sigma_decoder = nn.Sequential(
            nn.Linear(dec_in, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )
        self.ct_prob_weight = 0.001
        self._ct_scale_stat: Tensor | None = None
        self._ct_fused_stat: Tensor | None = None

    def _patch_stats(self, x: Tensor, mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        if "landau_edge_heat" in self.mode:
            pilot = ((dist >= 0) & (dist <= base_width)).to(dtype)
            pilot_weight = pilot * m_bdt
            pilot_denom = pilot_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pilot_mass = pilot_weight.sum(dim=-1, keepdim=True)
            pilot_total = pilot.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pilot_density = (pilot_mass / pilot_total).clamp(0.05, 1.0)
            pilot_mean = (pilot_weight * x_bdt).sum(dim=-1, keepdim=True) / pilot_denom
            pilot_second = (pilot_weight * x_bdt.square()).sum(dim=-1, keepdim=True) / pilot_denom
            pilot_var = (pilot_second - pilot_mean.square()).clamp_min(0.0)
            edge_stop = (1.0 / (1.0 + 4.0 * pilot_var)).clamp(0.0, 1.0)
            stretch = 1.0 + (pilot_density.rsqrt().clamp(1.0, 4.0) - 1.0) * edge_stop
            widths = widths * stretch
        elif "landau_heat" in self.mode or "density_heat" in self.mode:
            pilot = ((dist >= 0) & (dist <= base_width)).to(dtype)
            pilot_mass = (pilot * m_bdt).sum(dim=-1, keepdim=True)
            pilot_total = pilot.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pilot_density = (pilot_mass / pilot_total).clamp(0.05, 1.0)
            widths = widths * pilot_density.rsqrt().clamp(1.0, 4.0)

        if "landau_heat" in self.mode or "landau_edge_heat" in self.mode or "heat_kernel" in self.mode:
            z = dist.clamp_min(0.0) / widths.clamp_min(1e-4)
            kernel = torch.exp(-0.5 * z.square()) * causal
        elif "symmetric_kernel" in self.mode:
            kernel = torch.exp(-dist.abs() / widths.clamp_min(1e-4))
        elif "box_kernel" in self.mode:
            kernel = (dist.abs() <= widths).to(dtype)
        else:
            kernel = torch.exp(-dist.clamp_min(0.0) / widths.clamp_min(1e-4)) * causal

        weight = kernel * m_bdt
        denom = weight.sum(dim=-1).clamp_min(1e-6)
        density = weight.sum(dim=-1) / kernel.sum(dim=-1).clamp_min(1e-6)
        has_obs = (weight.sum(dim=-1) > 1e-6).to(dtype)

        mean = (weight * x_bdt).sum(dim=-1) / denom
        second = (weight * x_bdt.square()).sum(dim=-1) / denom
        var = (second - mean.square()).clamp_min(0.0)
        t_mean = (weight * t_hist).sum(dim=-1) / denom
        t_second = (weight * t_hist.square()).sum(dim=-1) / denom
        t_var = (t_second - t_mean.square()).clamp_min(1e-6)
        cov = (weight * (x_bdt - mean.unsqueeze(-1)) * (t_hist - t_mean.unsqueeze(-1))).sum(dim=-1) / denom
        slope = cov / t_var.clamp_min(1e-4)
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
        self._ct_scale_stat = scale_stat
        mid = min(1, self.n_scales - 1)
        stat = scale_stat[:, :, mid, :, :]
        patch_has_obs = has_obs.max(dim=2).values
        return stat, center_t, patch_has_obs

    def _make_tokens(self, stat: Tensor, center_t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self._ct_scale_stat is None:
            raise RuntimeError("CTPGNet _patch_stats must run before _make_tokens.")
        scale_stat = self._ct_scale_stat
        batch, n_vars, n_scales, n_patch, _ = scale_stat.shape
        var = self.var_emb.to(scale_stat.device, scale_stat.dtype).view(1, n_vars, 1, 1, self.d_model).expand(batch, -1, n_scales, n_patch, -1)
        te = self.time_enc(center_t).unsqueeze(2).expand(-1, -1, n_scales, -1, -1)
        base = torch.cat([scale_stat, te, var], dim=-1)
        token_k = self.ct_scale_op(base)

        if "single_scale" in self.mode:
            idx = min(1, n_scales - 1)
            alpha = torch.zeros(batch, n_vars, n_scales, n_patch, 1, device=scale_stat.device, dtype=scale_stat.dtype)
            alpha[:, :, idx, :, :] = 1.0
        elif "uniform_scale" in self.mode:
            alpha = torch.full((batch, n_vars, n_scales, n_patch, 1), 1.0 / n_scales, device=scale_stat.device, dtype=scale_stat.dtype)
        else:
            alpha = torch.softmax(self.ct_scale_score(base), dim=2)

        token = (alpha * token_k).sum(dim=2)
        fused_stat = (alpha * scale_stat).sum(dim=2)
        self._ct_fused_stat = fused_stat
        gate = alpha.max(dim=2).values.expand(-1, -1, -1, self.d_model)
        alias_gate = torch.sigmoid(self.alias_gate(torch.cat([fused_stat, token, token, torch.zeros_like(token)], dim=-1)))
        regime_gate = torch.zeros_like(token)
        return self.ct_norm(token + self.var_emb.to(token.device, token.dtype).view(1, n_vars, 1, self.d_model)), token, gate, alias_gate, regime_gate

    def _temporal_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        h = super()._temporal_graph(token, stat, alias_gate)
        if "no_stconv" in self.mode or "single_scale" in self.mode:
            return h
        msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        gate = torch.sigmoid(self.ct_st_gate(torch.cat([stat, h], dim=-1)))
        return self.ct_st_norm(h + gate * msg)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor = None,
        x_mask: Tensor = None,
        y: Tensor = None,
        y_mark: Tensor = None,
        y_mask: Tensor = None,
        **kwargs,
    ) -> dict:
        batch = x.size(0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x_mask = torch.nan_to_num(x_mask.to(dtype=x.dtype), nan=0.0).clamp(0.0, 1.0)
        if y is None:
            y = torch.zeros(batch, self.pred_len, self.n_vars, dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y)
        t = self._get_time(x_mark, x)
        y_t = self._get_time(y_mark, y) if y_mark is not None else _time(y.size(1), batch, x.device, x.dtype)

        stat, center_t, has_obs = self._patch_stats(x, x_mask, t)
        token, _, scale_gate, alias_gate, regime_gate = self._make_tokens(stat, center_t)
        stat_used = self._ct_fused_stat if self._ct_fused_stat is not None else stat
        h = self._temporal_graph(token, stat_used, alias_gate)

        score = self.pool_score(torch.cat([h, stat_used], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        enc = (weights * h).sum(dim=2)

        last_stat = stat_used[:, :, -1, :]
        global_state = enc.mean(dim=1, keepdim=True).expand(-1, self.n_vars, -1)
        enc_exp = enc.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        glob_exp = global_state.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        te_f = self.future_te(y_t).unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        tau = (y_t[:, :, 0] - t[:, -1, 0].view(batch, 1)).clamp_min(0.0)
        dec_mech = torch.stack(
            [
                tau.unsqueeze(1).expand(-1, self.n_vars, -1),
                last_stat[..., 0].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 2].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 3].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 5].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 6].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 9].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 10].unsqueeze(-1).expand(-1, -1, y.size(1)),
            ],
            dim=-1,
        )
        dec_in = torch.cat([enc_exp, glob_exp, te_f, dec_mech], dim=-1)
        pred = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)

        f_dim = -1 if self.configs.features == "MS" else 0
        out = {"pred": pred[:, -y.shape[1]:, f_dim:], "true": y[:, :, f_dim:], "mask": y_mask[:, :, f_dim:]}
        if self.training and "no_prob" not in self.mode:
            sigma = F.softplus(self.ct_sigma_decoder(dec_in).squeeze(-1).permute(0, 2, 1)) + 1e-3
            sigma = sigma[:, -y.shape[1]:, f_dim:]
            mask = y_mask[:, :, f_dim:].to(dtype=pred.dtype)
            residual = (out["pred"] - y[:, :, f_dim:]) * mask
            nll = 0.5 * (residual.square() / sigma.square() + 2.0 * torch.log(sigma)) * mask
            out["aux_loss"] = self.ct_prob_weight * nll.sum() / mask.sum().clamp_min(1.0)
        self.latest_diag = {
            "scale_gate_mean": float(scale_gate.mean().detach().cpu()),
            "patch_density": float(stat_used[..., 0].mean().detach().cpu()),
            "alias_gate_mean": float(alias_gate.mean().detach().cpu()),
            "regime_gate_mean": float(regime_gate.mean().detach().cpu()),
        }
        return out
