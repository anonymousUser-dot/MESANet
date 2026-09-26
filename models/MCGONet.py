import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from models.CAGNet import Model as CAGModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGModel):
    """Mechanism-Conditioned Graph Operator Network.

    MCGONet keeps CAGNet's successful admitted multiresolution patch operator
    and replaces the post-admission dependency layer with a mechanism-
    conditioned temporal/variable graph operator.  The key hypothesis is that
    wearable activity streams need two decisions: which temporal scale forms a
    patch token, and which variables should exchange information under the
    current observation mechanism.  CAGNet solves the first; MCGONet tests the
    second as a backbone-level operator.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        graph_dim = max(8, int(getattr(configs, "node_dim", 10)))
        self.mcg_state_q = MLP(12, self.hidden, graph_dim, self.dropout)
        self.mcg_state_k = MLP(12, self.hidden, graph_dim, self.dropout)
        self.mcg_edge_mix = MLP(12 + self.d_model, self.hidden, 1, self.dropout)
        self.mcg_prior_weight = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.mcg_gate = MLP(12 + self.d_model * 2, self.hidden, self.d_model, self.dropout)
        self.mcg_norm = nn.LayerNorm(self.d_model)
        self.mcg_st_gate = MLP(12 + self.d_model * 2, self.hidden, self.d_model, self.dropout)
        self.mcg_st_norm = nn.LayerNorm(self.d_model)
        last = self.mcg_edge_mix.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, -1.0)

    def _mechanism_affinity_prior(self, stat: Tensor) -> Tensor:
        density = stat[..., 0].permute(0, 2, 1).clamp(0.0, 1.0)
        mean = stat[..., 2].permute(0, 2, 1)
        var = stat[..., 3].permute(0, 2, 1).clamp_min(0.0)
        span = stat[..., 9].permute(0, 2, 1).clamp(0.0, 1.0)
        recency = stat[..., 10].permute(0, 2, 1).clamp(0.0, 1.0)

        reliability = torch.sqrt((density.unsqueeze(-1) * density.unsqueeze(-2)).clamp_min(1e-6))
        value_dist = (mean.unsqueeze(-1) - mean.unsqueeze(-2)).square()
        value_scale = var.unsqueeze(-1) + var.unsqueeze(-2) + 1.0e-3
        mechanism_dist = (density.unsqueeze(-1) - density.unsqueeze(-2)).abs()
        mechanism_dist = mechanism_dist + 0.5 * (recency.unsqueeze(-1) - recency.unsqueeze(-2)).abs()
        mechanism_dist = mechanism_dist + 0.25 * (span.unsqueeze(-1) - span.unsqueeze(-2)).abs()
        affinity = reliability
        affinity = affinity * torch.exp(-(value_dist / value_scale).clamp_max(8.0))
        affinity = affinity * torch.exp(-mechanism_dist.clamp_max(8.0))
        return torch.log(affinity.clamp_min(1.0e-4))

    def _mechanism_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        batch, n_vars, n_patch, dim = token.shape
        pos = self.temporal_pos[:, :n_patch, :].to(token.device, token.dtype)
        h = token.reshape(batch * n_vars, n_patch, dim) + pos
        h = self.temporal(h).view(batch, n_vars, n_patch, dim)
        if "no_graph" in self.mode:
            return h
        if "no_mcg" in self.mode or "content_graph" in self.mode:
            return AMPGModel._temporal_graph(self, token, stat, alias_gate)

        q = self.graph_q(h).permute(0, 2, 1, 3)
        k = self.graph_k(h).permute(0, 2, 1, 3)
        content_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))

        edge_logits = content_logits
        if "no_state_edge" not in self.mode:
            sq = self.mcg_state_q(stat).permute(0, 2, 1, 3)
            sk = self.mcg_state_k(stat).permute(0, 2, 1, 3)
            state_logits = torch.matmul(sq, sk.transpose(-1, -2)) / math.sqrt(sq.size(-1))
            mix = torch.sigmoid(self.mcg_edge_mix(torch.cat([stat, h], dim=-1))).permute(0, 2, 1, 3)
            if "closed_edge" in self.mode or "tiny_edge" in self.mode:
                mix = 0.5 * mix
            edge_logits = edge_logits + mix * state_logits

        if "no_prior" not in self.mode:
            prior = self._mechanism_affinity_prior(stat)
            weight = F.softplus(self.mcg_prior_weight)
            if "strong_prior" in self.mode:
                weight = 2.0 * weight
            elif "weak_prior" in self.mode:
                weight = 0.5 * weight
            edge_logits = edge_logits + weight * prior

        attn = torch.softmax(edge_logits, dim=-1)
        msg = torch.einsum("bpij,bjpd->bipd", attn, h)
        msg = self.graph_msg(msg)
        gate = torch.sigmoid(self.mcg_gate(torch.cat([stat, h, msg], dim=-1)))
        if alias_gate is not None:
            gate = gate * (0.25 + 0.75 * alias_gate.mean(dim=-1, keepdim=True).clamp(0.0, 1.0))
        h = self.mcg_norm(h + gate * msg)

        if "no_stconv" not in self.mode:
            st_msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            st_gate = torch.sigmoid(self.mcg_st_gate(torch.cat([stat, h, st_msg], dim=-1)))
            if "closed_st" in self.mode:
                st_gate = 0.5 * st_gate
            h = self.mcg_st_norm(h + st_gate * st_msg)
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

        h = self._mechanism_graph(token, stat, fixed_alias)
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
            sigma = F.softplus(self.ct_sigma_decoder(dec_in).squeeze(-1).permute(0, 2, 1)) + 1e-3
            sigma = sigma[:, -y.shape[1]:, f_dim:]
            mask = y_mask[:, :, f_dim:].to(dtype=pred.dtype)
            residual = (out["pred"] - y[:, :, f_dim:]) * mask
            nll = 0.5 * (residual.square() / sigma.square() + 2.0 * torch.log(sigma)) * mask
            out["aux_loss"] = self.ct_prob_weight * nll.sum() / mask.sum().clamp_min(1.0)
        self.latest_diag = {
            "admit_mean": float(admit.mean().detach().cpu()),
            "patch_density": float(stat[..., 0].mean().detach().cpu()),
            "alias_gate_mean": float(fixed_alias.mean().detach().cpu()),
        }
        return out
