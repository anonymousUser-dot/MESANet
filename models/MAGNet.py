import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from models.CTPGNet import Model as CTPGModel
from utils.ExpConfigs import ExpConfigs


class Model(CTPGModel):
    """Mechanism-Adaptive Graph Network.

    MAGNet is an end-to-end IMTS forecasting backbone for wearable streams.
    It treats the deployment-visible mechanism state of a patch, including
    missingness density, arrival span, recency, and local motion statistics, as
    the state that determines how observations should be resolved. The state
    modulates continuous-time patch scales, variable-graph edges, and the
    mechanism/value mixture before the decoder sees any forecast.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.mode = str(getattr(configs, "ablation_name", "") or "").lower()
        self.mech_emb = MLP(12, self.hidden, self.d_model, self.dropout)
        self.state_scale_bias = MLP(12 + self.d_model, self.hidden, self.n_scales, self.dropout)
        self.state_token_gate = MLP(12 + self.d_model * 2, self.hidden, self.d_model, self.dropout)
        self.state_edge_q = nn.Linear(self.d_model, max(8, int(getattr(configs, "node_dim", 10))))
        self.state_edge_k = nn.Linear(self.d_model, max(8, int(getattr(configs, "node_dim", 10))))
        self.state_edge_mix = MLP(12 + self.d_model * 2, self.hidden, 1, self.dropout)
        self.laplace_edge_weight = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.state_decoder_bias = MLP(12 + self.d_model, self.hidden, self.d_model, self.dropout)
        self.mag_norm = nn.LayerNorm(self.d_model)
        self._mag_fixed_stat: Tensor | None = None
        self._mag_mech_token: Tensor | None = None
        self._time_irregularity: Tensor | None = None

    def _get_fixed_state(self, x: Tensor, x_mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stat, center_t, has_obs = AMPGModel._patch_stats(self, x, x_mask, t)
        return stat, center_t, has_obs

    def _make_tokens(self, stat: Tensor, center_t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self._ct_scale_stat is None:
            raise RuntimeError("MAGNet _patch_stats must run before _make_tokens.")
        scale_stat = self._ct_scale_stat
        batch, n_vars, n_scales, n_patch, _ = scale_stat.shape
        fixed_stat = self._mag_fixed_stat
        if fixed_stat is None:
            fixed_stat = scale_stat[:, :, min(1, n_scales - 1), :, :]
        fixed_stat = fixed_stat[:, :, :n_patch, :]

        var = self.var_emb.to(scale_stat.device, scale_stat.dtype).view(1, n_vars, 1, 1, self.d_model)
        var = var.expand(batch, -1, n_scales, n_patch, -1)
        te = self.time_enc(center_t).unsqueeze(2).expand(-1, -1, n_scales, -1, -1)
        base = torch.cat([scale_stat, te, var], dim=-1)
        token_k = self.ct_scale_op(base)

        base_logits = self.ct_scale_score(base).squeeze(-1)
        mech = self.mech_emb(fixed_stat)
        if "no_state_scale" in self.mode:
            scale_logits = base_logits
        else:
            state_bias = self.state_scale_bias(torch.cat([fixed_stat, mech], dim=-1)).permute(0, 1, 3, 2)
            if "gap_guarded_scale" in self.mode and self._time_irregularity is not None:
                gap_gate = torch.sigmoid(20.0 * (self._time_irregularity.to(state_bias.dtype) - 0.05))
                state_bias = state_bias * gap_gate.view(batch, 1, 1, 1)
            scale_logits = base_logits + state_bias
        if "uniform_scale" in self.mode:
            alpha = torch.full((batch, n_vars, n_scales, n_patch, 1), 1.0 / n_scales, device=scale_stat.device, dtype=scale_stat.dtype)
        else:
            alpha = torch.softmax(scale_logits, dim=2).unsqueeze(-1)

        ct_token = (alpha * token_k).sum(dim=2)
        fused_stat = (alpha * scale_stat).sum(dim=2)
        if "no_mech_token" in self.mode:
            token = ct_token
            token_gate = torch.zeros_like(ct_token)
        else:
            token_gate = torch.sigmoid(self.state_token_gate(torch.cat([fused_stat, ct_token, mech], dim=-1)))
            if "hard_mech_token" in self.mode:
                token_gate = (token_gate > 0.5).to(token_gate.dtype)
            token = ct_token + token_gate * (mech - ct_token)
        token = self.mag_norm(token + self.var_emb.to(token.device, token.dtype).view(1, n_vars, 1, self.d_model))
        self._ct_fused_stat = fused_stat
        self._mag_mech_token = mech
        alias_gate = torch.sigmoid(self.alias_gate(torch.cat([fused_stat, token, mech, (token - mech).abs()], dim=-1)))
        regime_gate = token_gate
        scale_gate = alpha.max(dim=2).values.expand(-1, -1, -1, self.d_model)
        return token, ct_token, scale_gate, alias_gate, regime_gate

    def _temporal_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        batch, n_vars, n_patch, dim = token.shape
        pos = self.temporal_pos[:, :n_patch, :].to(token.device, token.dtype)
        h = token.reshape(batch * n_vars, n_patch, dim) + pos
        h = self.temporal(h).view(batch, n_vars, n_patch, dim)
        if "no_graph" in self.mode:
            return h

        mech = self._mag_mech_token
        if mech is None or "no_state_edge" in self.mode:
            return AMPGModel._temporal_graph(self, token, stat, alias_gate)

        q = self.graph_q(h).permute(0, 2, 1, 3)
        k = self.graph_k(h).permute(0, 2, 1, 3)
        content_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        mq = self.state_edge_q(mech).permute(0, 2, 1, 3)
        mk = self.state_edge_k(mech).permute(0, 2, 1, 3)
        state_logits = torch.matmul(mq, mk.transpose(-1, -2)) / math.sqrt(mq.size(-1))
        mix = torch.sigmoid(self.state_edge_mix(torch.cat([stat, h, mech], dim=-1))).permute(0, 2, 1, 3)
        edge_logits = content_logits + mix * state_logits
        if "laplacian_edge" in self.mode:
            density = stat[..., 0].permute(0, 2, 1).clamp(0.0, 1.0)
            mean = stat[..., 2].permute(0, 2, 1)
            var = stat[..., 3].permute(0, 2, 1).clamp_min(0.0)
            recency = stat[..., 10].permute(0, 2, 1).clamp(0.0, 1.0)
            reliability = torch.sqrt((density.unsqueeze(-1) * density.unsqueeze(-2)).clamp_min(1e-6))
            value_dist = (mean.unsqueeze(-1) - mean.unsqueeze(-2)).square()
            value_scale = var.unsqueeze(-1) + var.unsqueeze(-2) + 1e-3
            mechanism_dist = (density.unsqueeze(-1) - density.unsqueeze(-2)).abs()
            mechanism_dist = mechanism_dist + 0.25 * (recency.unsqueeze(-1) - recency.unsqueeze(-2)).abs()
            affinity = reliability * torch.exp(-(value_dist / value_scale).clamp_max(8.0)) * torch.exp(-mechanism_dist)
            prior_logits = torch.log(affinity.clamp_min(1e-4))
            edge_logits = edge_logits + F.softplus(self.laplace_edge_weight) * prior_logits
        attn = torch.softmax(edge_logits, dim=-1)
        msg = torch.einsum("bpij,bjpd->bipd", attn, h)
        msg = self.graph_msg(msg)
        gate = torch.sigmoid(self.graph_gate(torch.cat([h, msg, stat], dim=-1)))
        if alias_gate is not None:
            gate = gate * (0.25 + 0.75 * alias_gate.mean(dim=-1, keepdim=True).clamp(0.0, 1.0))
        h = self.graph_norm(h + gate * msg)
        if "no_stconv" not in self.mode:
            st_msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            st_gate = torch.sigmoid(self.ct_st_gate(torch.cat([stat, h], dim=-1)))
            h = self.ct_st_norm(h + st_gate * st_msg)
        return h

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

        fixed_stat, _, fixed_obs = self._get_fixed_state(x, x_mask, t)
        self._mag_fixed_stat = fixed_stat
        stat, center_t, ct_obs = CTPGModel._patch_stats(self, x, x_mask, t)
        token, _, scale_gate, alias_gate, regime_gate = self._make_tokens(stat, center_t)
        stat_used = self._ct_fused_stat if self._ct_fused_stat is not None else stat
        has_obs = torch.maximum(ct_obs, fixed_obs[:, :, : ct_obs.size(-1)])
        h = self._temporal_graph(token, stat_used, alias_gate)

        score = self.pool_score(torch.cat([h, stat_used], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        enc = (weights * h).sum(dim=2)
        last_stat = stat_used[:, :, -1, :]
        if self._mag_mech_token is not None and "no_decoder_state" not in self.mode:
            last_mech = self._mag_mech_token[:, :, -1, :]
            enc = self.mag_norm(enc + self.state_decoder_bias(torch.cat([last_stat, last_mech], dim=-1)))

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
