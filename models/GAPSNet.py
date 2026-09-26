import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from utils.ExpConfigs import ExpConfigs


class Model(AMPGModel):
    """Gap-Adaptive Patch Spatio-Temporal Prior Network.

    GAPS-Net keeps the deployment-valid patch statistics used by AMPG/MRPS, but
    changes the backbone principle: time gaps define a predictive coordinate
    system, patches are fused at multiple temporal scales, and the token is
    mixed by a small spatio-temporal convolution before graph transfer.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
        stat_dim = 12
        self.gap_state = MLP(stat_dim + te_dim, self.hidden, self.d_model, self.dropout)
        self.gap_to_local = nn.Linear(self.d_model, self.d_model)
        self.gap_to_integral = nn.Linear(self.d_model, self.d_model)
        self.scale_gate = nn.Linear(stat_dim + self.d_model, 3)
        self.scale_norm = nn.LayerNorm(self.d_model)

        self.st_conv = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
        )
        self.st_gate = MLP(stat_dim + self.d_model, self.hidden, self.d_model, self.dropout)
        self.st_norm = nn.LayerNorm(self.d_model)

        dec_in = self.d_model * 2 + te_dim + 8
        self.sigma_decoder = nn.Sequential(
            nn.Linear(dec_in, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )
        self.prob_weight = 0.002

    def _smooth_patch(self, token: Tensor, kernel: int) -> Tensor:
        if kernel <= 1 or token.size(2) <= 1:
            return token
        bsz, n_vars, n_patch, dim = token.shape
        h = token.reshape(bsz * n_vars, n_patch, dim).transpose(1, 2)
        h = F.avg_pool1d(h, kernel_size=kernel, stride=1, padding=kernel // 2)
        if h.size(-1) != n_patch:
            h = h[..., :n_patch]
        return h.transpose(1, 2).reshape(bsz, n_vars, n_patch, dim)

    def _make_tokens(self, stat: Tensor, center_t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, n_vars, n_patch, _ = stat.shape
        var = self.var_emb.to(stat.device, stat.dtype).view(1, n_vars, 1, self.d_model).expand(batch, -1, n_patch, -1)
        te = self.time_enc(center_t)
        base = torch.cat([stat, te, var], dim=-1)

        local_base = self._branch_base(stat, te, var, self.local_stat_cols)
        integral_base = self._branch_base(stat, te, var, self.integral_stat_cols)
        local = self.local_op(local_base)
        integral = self.integral_op(integral_base)

        if "no_gap" not in self.mode:
            gap = self.gap_state(torch.cat([stat, te], dim=-1))
            local = local + self.gap_to_local(gap)
            integral = integral + self.gap_to_integral(gap)
        else:
            gap = torch.zeros_like(local)

        closed_gate = torch.sigmoid(self.mechanism_gate(torch.cat([base, local, integral], dim=-1)))
        token = (1.0 - closed_gate) * local + closed_gate * integral

        if "no_adaptive_patch" not in self.mode and "gap_only" not in self.mode:
            mid = self._smooth_patch(token, 3)
            wide = self._smooth_patch(token, 5)
            alpha = torch.softmax(self.scale_gate(torch.cat([stat, gap], dim=-1)), dim=-1).unsqueeze(-1)
            token = alpha[..., 0, :] * token + alpha[..., 1, :] * mid + alpha[..., 2, :] * wide
            token = self.scale_norm(token)

        alias_gate = torch.sigmoid(self.alias_gate(torch.cat([stat, local, integral, (local - integral).abs()], dim=-1)))
        regime_gate = torch.zeros_like(local)
        return self.patch_norm(token + var), local, closed_gate, alias_gate, regime_gate

    def _temporal_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        h = super()._temporal_graph(token, stat, alias_gate)
        if "no_stconv" in self.mode or "gap_only" in self.mode or "adaptive_only" in self.mode:
            return h
        z = h.permute(0, 3, 1, 2)
        msg = self.st_conv(z).permute(0, 2, 3, 1)
        gate = torch.sigmoid(self.st_gate(torch.cat([stat, h], dim=-1)))
        return self.st_norm(h + gate * msg)

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
        batch, _, _ = x.shape
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
        token, local, mech_gate, alias_gate, regime_gate = self._make_tokens(stat, center_t)
        h = self._temporal_graph(token, stat, alias_gate)

        score = self.pool_score(torch.cat([h, stat], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        enc = (weights * h).sum(dim=2)

        last_stat = stat[:, :, -1, :]
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
            sigma = F.softplus(self.sigma_decoder(dec_in).squeeze(-1).permute(0, 2, 1)) + 1e-3
            sigma = sigma[:, -y.shape[1]:, f_dim:]
            target = y[:, :, f_dim:]
            mask = y_mask[:, :, f_dim:].to(dtype=pred.dtype)
            residual = (out["pred"] - target) * mask
            nll = 0.5 * (residual.square() / sigma.square() + 2.0 * torch.log(sigma)) * mask
            denom = mask.sum().clamp_min(1.0)
            out["aux_loss"] = self.prob_weight * nll.sum() / denom
        self.latest_diag = {
            "mechanism_gate_mean": float(mech_gate.mean().detach().cpu()),
            "patch_density": float(stat[..., 0].mean().detach().cpu()),
            "alias_gate_mean": float(alias_gate.mean().detach().cpu()),
            "regime_gate_mean": float(regime_gate.mean().detach().cpu()),
        }
        return out
