import torch
import torch.nn as nn
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from models.CAGNet import Model as CAGModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGModel):
    """Arrival-State Level-Trend Network.

    ALT-Net is a native decoder-level IMTS backbone. It preserves CAGNet's
    admitted patch encoder but changes forecast formation: each future variable
    is predicted by a learned neural path and a deployment-valid level/trend
    anchor derived from the final admitted mechanism state. A closed gate blends
    the two inside the decoder. This tests whether wearable accuracy is limited
    less by token mixing and more by whether the forecast head respects the
    latest observed level and local trend.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
        dec_in = self.d_model * 2 + te_dim + 8
        self.anchor_gate = MLP(dec_in + 3, self.hidden, 1, self.dropout)
        self.anchor_norm = nn.LayerNorm(3)
        last = self.anchor_gate.net[-1]
        if isinstance(last, nn.Linear):
            bias = -2.0
            if "open_anchor" in self.mode:
                bias = -0.5
            elif "very_closed_anchor" in self.mode:
                bias = -3.0
            nn.init.constant_(last.bias, bias)

    def _anchor_strength(self) -> float:
        if "weak_anchor" in self.mode:
            return 0.25
        if "strong_anchor" in self.mode:
            return 1.00
        return 0.50

    def _anchor_bound_factor(self) -> float:
        if "tight_anchor" in self.mode:
            return 0.50
        if "loose_anchor" in self.mode:
            return 2.00
        return 1.00

    def _level_trend_anchor(self, last_stat: Tensor, pred_steps: int) -> tuple[Tensor, Tensor, Tensor]:
        dtype = last_stat.dtype
        device = last_stat.device
        last_x = last_stat[..., 5].unsqueeze(-1)
        slope = last_stat[..., 6].unsqueeze(-1)
        var = last_stat[..., 3].clamp_min(0.0).unsqueeze(-1)
        span = last_stat[..., 9].clamp_min(1.0e-3).unsqueeze(-1)
        horizon = torch.arange(1, pred_steps + 1, device=device, dtype=dtype).view(1, 1, pred_steps)
        horizon = horizon / max(pred_steps, 1) * span
        if "no_slope_anchor" in self.mode:
            raw_delta = torch.zeros_like(horizon)
        else:
            raw_delta = self._anchor_strength() * slope * horizon
        bound = self._anchor_bound_factor() * (torch.sqrt(var + 1.0e-4) + 0.05)
        delta = bound * torch.tanh(raw_delta / bound.clamp_min(1.0e-4))
        anchor = last_x + delta
        return anchor, delta, bound.expand_as(delta)

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
        if t.size(1) > 1:
            dt = (t[:, 1:, 0] - t[:, :-1, 0]).clamp_min(0.0)
            dt_mean = dt.mean(dim=1)
            dt_std = dt.std(dim=1, unbiased=False)
            self._time_irregularity = (dt_std / dt_mean.clamp_min(1e-6)).clamp(0.0, 1.0)
        else:
            self._time_irregularity = torch.zeros(batch, device=x.device, dtype=x.dtype)
        y_t = self._get_time(y_mark, y) if y_mark is not None else _time(y.size(1), batch, x.device, x.dtype)

        fixed_token, fixed_stat, fixed_obs, fixed_alias = self._fixed_branch(x, x_mask, t)
        scale_token, scale_stat, scale_obs = self._scale_branch(x, x_mask, t, fixed_stat)
        n_patch = min(fixed_token.size(2), scale_token.size(2))
        fixed_token = fixed_token[:, :, :n_patch, :]
        fixed_stat = fixed_stat[:, :, :n_patch, :]
        fixed_obs = fixed_obs[:, :, :n_patch]
        fixed_alias = fixed_alias[:, :, :n_patch, :]
        scale_token = scale_token[:, :, :n_patch, :]
        scale_stat = scale_stat[:, :, :n_patch, :]
        scale_obs = scale_obs[:, :, :n_patch]
        token, stat, admit = self._admit(fixed_token, fixed_stat, scale_token, scale_stat)
        has_obs = torch.maximum(fixed_obs, scale_obs)

        h = AMPGModel._temporal_graph(self, token, stat, fixed_alias)
        if "no_stconv" not in self.mode:
            msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            gate = torch.sigmoid(self.ct_st_gate(torch.cat([stat, h], dim=-1)))
            h = self.ct_st_norm(h + gate * msg)

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
        neural = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)
        anchor, delta, bound = self._level_trend_anchor(last_stat, y.size(1))
        anchor_feat = torch.stack([anchor, delta, bound], dim=-1)
        anchor_feat = self.anchor_norm(anchor_feat)
        anchor_gate = torch.sigmoid(self.anchor_gate(torch.cat([dec_in, anchor_feat], dim=-1))).squeeze(-1)
        if "mean_anchor" in self.mode:
            anchor_gate = torch.full_like(anchor_gate, 0.5)
        if "anchor_only" in self.mode:
            pred_bvt = anchor
            anchor_gate = torch.ones_like(anchor_gate)
        elif "no_anchor" in self.mode:
            pred_bvt = neural.permute(0, 2, 1)
            anchor_gate = torch.zeros_like(anchor_gate)
        else:
            pred_bvt = neural.permute(0, 2, 1) + anchor_gate * (anchor - neural.permute(0, 2, 1))
        pred = pred_bvt.permute(0, 2, 1)

        f_dim = -1 if self.configs.features == "MS" else 0
        out = {"pred": pred[:, -y.shape[1]:, f_dim:], "true": y[:, :, f_dim:], "mask": y_mask[:, :, f_dim:]}
        self.latest_diag = {
            "admit_mean": float(admit.mean().detach().cpu()),
            "anchor_gate_mean": float(anchor_gate.mean().detach().cpu()),
            "anchor_delta_abs": float(delta.abs().mean().detach().cpu()),
            "patch_density": float(stat[..., 0].mean().detach().cpu()),
            "alias_gate_mean": float(fixed_alias.mean().detach().cpu()),
        }
        return out
