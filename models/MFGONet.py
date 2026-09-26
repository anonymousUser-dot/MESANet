import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from models.MCGONet import Model as MCGOModel
from utils.ExpConfigs import ExpConfigs


class Model(MCGOModel):
    """Metric-aligned Forecast-Gated Operator Network.

    MFGO-Net keeps the local CAG amplitude coordinate as the default forecast
    path and moves cross-variable mechanism information to a closed forecast
    gate. The graph branch can propose a contextual forecast, but it is admitted
    only at the prediction head and its delta is bounded by the observed patch
    scale. This directly targets the pattern found in v493-v495: graph/context
    operators often improve absolute-error smoothness but can hurt MSE by
    rewriting transition amplitudes inside the encoder.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
        gate_in = self.d_model * 3 + te_dim + 8
        self.mfgo_forecast_gate = MLP(gate_in, self.hidden, 1, self.dropout)
        self.mfgo_delta_scale = MLP(12 + self.d_model * 2, self.hidden, 1, self.dropout)
        self.mfgo_amp_norm = nn.LayerNorm(self.d_model)
        gate_last = self.mfgo_forecast_gate.net[-1]
        if isinstance(gate_last, nn.Linear):
            bias = -2.0
            if "open_forecast_gate" in self.mode:
                bias = -0.5
            elif "very_closed_forecast_gate" in self.mode:
                bias = -3.0
            nn.init.constant_(gate_last.bias, bias)
        scale_last = self.mfgo_delta_scale.net[-1]
        if isinstance(scale_last, nn.Linear):
            nn.init.zeros_(scale_last.weight)
            nn.init.zeros_(scale_last.bias)

    def _content_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        h = AMPGModel._temporal_graph(self, token, stat, alias_gate)
        if "no_amp_stconv" not in self.mode and "no_stconv" not in self.mode:
            msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            gate = torch.sigmoid(self.ct_st_gate(torch.cat([stat, h], dim=-1)))
            h = self.ct_st_norm(h + gate * msg)
        return self.mfgo_amp_norm(h)

    def _pool(self, h: Tensor, stat: Tensor, has_obs: Tensor) -> Tensor:
        score = self.pool_score(torch.cat([h, stat], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1.0e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        return (weights * h).sum(dim=2)

    def _decoder_state(self, enc: Tensor, stat: Tensor, t: Tensor, y: Tensor, y_t: Tensor) -> tuple[Tensor, Tensor]:
        batch = enc.size(0)
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
        return torch.cat([enc_exp, glob_exp, te_f, dec_mech], dim=-1), dec_mech

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
            self._time_irregularity = (dt_std / dt_mean.clamp_min(1.0e-6)).clamp(0.0, 1.0)
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

        amp_h = self._content_graph(token, stat, fixed_alias)
        graph_h = MCGOModel._mechanism_graph(self, token, stat, fixed_alias)
        amp_enc = self._pool(amp_h, stat, has_obs)
        graph_enc = self._pool(graph_h, stat, has_obs)

        amp_dec, dec_mech = self._decoder_state(amp_enc, stat, t, y, y_t)
        graph_dec, _ = self._decoder_state(graph_enc, stat, t, y, y_t)
        amp_pred = self.decoder(amp_dec).squeeze(-1)
        graph_pred = self.decoder(graph_dec).squeeze(-1)

        te_f = self.future_te(y_t).unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        amp_exp = amp_enc.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        graph_exp = graph_enc.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        gate_in = torch.cat([amp_exp, graph_exp, (amp_exp - graph_exp).abs(), te_f, dec_mech], dim=-1)
        gate = torch.sigmoid(self.mfgo_forecast_gate(gate_in)).squeeze(-1)
        if "amp_only" in self.mode:
            gate = torch.zeros_like(gate)
        elif "graph_only" in self.mode:
            gate = torch.ones_like(gate)
        elif "mean_forecast_gate" in self.mode:
            gate = torch.full_like(gate, 0.5)

        delta = graph_pred - amp_pred
        if "unbounded_delta" not in self.mode:
            last_stat = stat[:, :, -1, :]
            obs_scale = torch.sqrt(last_stat[..., 3].clamp_min(0.0) + 1.0e-4).unsqueeze(-1)
            learned = F.softplus(self.mfgo_delta_scale(torch.cat([last_stat, amp_enc, graph_enc], dim=-1))).squeeze(-1)
            bound = (0.25 + learned).unsqueeze(-1) * obs_scale.clamp_min(0.05)
            if "tight_delta" in self.mode:
                bound = 0.5 * bound
            elif "loose_delta" in self.mode:
                bound = 2.0 * bound
            delta = torch.tanh(delta / bound.clamp_min(1.0e-4)) * bound
        pred = (amp_pred + gate * delta).permute(0, 2, 1)

        f_dim = -1 if self.configs.features == "MS" else 0
        out = {"pred": pred[:, -y.shape[1]:, f_dim:], "true": y[:, :, f_dim:], "mask": y_mask[:, :, f_dim:]}
        if self.training and "no_prob" not in self.mode:
            sigma = F.softplus(self.ct_sigma_decoder(amp_dec).squeeze(-1).permute(0, 2, 1)) + 1.0e-3
            sigma = sigma[:, -y.shape[1]:, f_dim:]
            mask = y_mask[:, :, f_dim:].to(dtype=pred.dtype)
            residual = (out["pred"] - y[:, :, f_dim:]) * mask
            nll = 0.5 * (residual.square() / sigma.square() + 2.0 * torch.log(sigma)) * mask
            out["aux_loss"] = self.ct_prob_weight * nll.sum() / mask.sum().clamp_min(1.0)
        self.latest_diag = {
            "admit_mean": float(admit.mean().detach().cpu()),
            "forecast_gate_mean": float(gate.mean().detach().cpu()),
            "patch_density": float(stat[..., 0].mean().detach().cpu()),
            "alias_gate_mean": float(fixed_alias.mean().detach().cpu()),
        }
        return out
