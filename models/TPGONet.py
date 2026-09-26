import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel
from models.MCGONet import Model as MCGOModel
from utils.ExpConfigs import ExpConfigs


class Model(MCGOModel):
    """Transition-Preserving Graph Operator Network.

    TPGO-Net keeps CAGNet's level-preserving admitted patch representation and
    replaces unrestricted mechanism graph exchange with a transition-preserving
    operator. Wearable forecasting errors are often dominated by activity
    boundaries: a graph message that is harmless in a stationary segment can
    blur amplitude jumps and hurt MSE at transitions. TPGO therefore uses the
    deployment-visible patch state to (i) reduce scale admission at high local
    transition energy and (ii) exchange cross-variable messages mainly between
    variables whose transition phase is compatible.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.tpg_transition_gate = MLP(12 + self.d_model * 2, self.hidden, self.d_model, self.dropout)
        self.tpg_edge_weight = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        last = self.tpg_transition_gate.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, -1.0)

    @staticmethod
    def _transition_energy(stat: Tensor) -> Tensor:
        var = stat[..., 3].clamp_min(0.0)
        jump = (stat[..., 5] - stat[..., 4]).abs()
        slope = stat[..., 6].abs()
        span = stat[..., 9].clamp_min(0.0)
        raw = var + 0.25 * jump + 0.025 * slope * span
        return (raw / (raw + 1.0)).clamp(0.0, 1.0)

    @staticmethod
    def _transition_phase(stat: Tensor) -> Tensor:
        var = stat[..., 3].clamp_min(0.0)
        jump = stat[..., 5] - stat[..., 4]
        slope = stat[..., 6]
        denom = torch.sqrt(var + 1.0e-4) + jump.abs() + 0.05
        return torch.tanh((jump + 0.05 * slope) / denom)

    def _admit(self, fixed_token: Tensor, fixed_stat: Tensor, scale_token: Tensor, scale_stat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        token, stat, scalar = super()._admit(fixed_token, fixed_stat, scale_token, scale_stat)
        if "no_transition_admit_guard" in self.mode:
            return token, stat, scalar
        strength = 0.55
        if "weak_transition_guard" in self.mode:
            strength = 0.30
        elif "strong_transition_guard" in self.mode:
            strength = 0.80
        transition = self._transition_energy(fixed_stat).unsqueeze(-1)
        guard = (1.0 - strength * transition).clamp(0.10, 1.0)
        guarded = scalar * guard
        token = self.admit_norm(fixed_token + guarded * (scale_token - fixed_token))
        stat = fixed_stat + guarded * (scale_stat - fixed_stat)
        return token, stat, guarded

    def _mechanism_affinity_prior(self, stat: Tensor) -> Tensor:
        base = super()._mechanism_affinity_prior(stat).exp()
        density = stat[..., 0].permute(0, 2, 1).clamp(0.0, 1.0)
        recency = stat[..., 10].permute(0, 2, 1).clamp(0.0, 1.0)
        energy = self._transition_energy(stat).permute(0, 2, 1)
        phase = self._transition_phase(stat).permute(0, 2, 1)

        reliability = torch.sqrt((density.unsqueeze(-1) * density.unsqueeze(-2)).clamp_min(1.0e-6))
        stale_match = torch.exp(-0.5 * (recency.unsqueeze(-1) - recency.unsqueeze(-2)).abs())
        phase_match = torch.exp(-1.5 * (phase.unsqueeze(-1) - phase.unsqueeze(-2)).abs())
        stable_pair = torch.sqrt(((1.0 - energy).unsqueeze(-1) * (1.0 - energy).unsqueeze(-2)).clamp_min(1.0e-6))
        transition_pair = torch.sqrt((energy.unsqueeze(-1) * energy.unsqueeze(-2)).clamp_min(1.0e-6))
        transition_prior = reliability * stale_match * (0.35 * stable_pair + phase_match * transition_pair)
        prior = torch.sqrt(base.clamp_min(1.0e-6) * transition_prior.clamp_min(1.0e-6))
        return torch.log(prior.clamp_min(1.0e-4))

    def _mechanism_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        batch, n_vars, n_patch, dim = token.shape
        pos = self.temporal_pos[:, :n_patch, :].to(token.device, token.dtype)
        h = token.reshape(batch * n_vars, n_patch, dim) + pos
        h = self.temporal(h).view(batch, n_vars, n_patch, dim)
        if "no_graph" in self.mode:
            return h
        if "content_graph" in self.mode:
            return AMPGModel._temporal_graph(self, token, stat, alias_gate)

        q = self.graph_q(h).permute(0, 2, 1, 3)
        k = self.graph_k(h).permute(0, 2, 1, 3)
        edge_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))

        if "no_state_edge" not in self.mode:
            sq = self.mcg_state_q(stat).permute(0, 2, 1, 3)
            sk = self.mcg_state_k(stat).permute(0, 2, 1, 3)
            state_logits = torch.matmul(sq, sk.transpose(-1, -2)) / math.sqrt(sq.size(-1))
            mix = torch.sigmoid(self.mcg_edge_mix(torch.cat([stat, h], dim=-1))).permute(0, 2, 1, 3)
            if "closed_edge" in self.mode or "tiny_edge" in self.mode:
                mix = 0.5 * mix
            edge_logits = edge_logits + mix * state_logits

        if "no_transition_prior" not in self.mode:
            weight = F.softplus(self.tpg_edge_weight)
            if "strong_transition_prior" in self.mode:
                weight = 2.0 * weight
            elif "weak_transition_prior" in self.mode:
                weight = 0.5 * weight
            edge_logits = edge_logits + weight * self._mechanism_affinity_prior(stat)
        elif "no_prior" not in self.mode:
            edge_logits = edge_logits + F.softplus(self.mcg_prior_weight) * super()._mechanism_affinity_prior(stat)

        attn = torch.softmax(edge_logits, dim=-1)
        msg = torch.einsum("bpij,bjpd->bipd", attn, h)
        msg = self.graph_msg(msg)

        energy = self._transition_energy(stat)
        phase = self._transition_phase(stat)
        phase_bp = phase.permute(0, 2, 1)
        phase_match = torch.exp(-1.5 * (phase_bp.unsqueeze(-1) - phase_bp.unsqueeze(-2)).abs())
        transition_sync = (attn * phase_match).sum(dim=-1).permute(0, 2, 1).unsqueeze(-1)
        base_gate = torch.sigmoid(self.mcg_gate(torch.cat([stat, h, msg], dim=-1)))
        trans_gate = torch.sigmoid(self.tpg_transition_gate(torch.cat([stat, h, msg], dim=-1)))
        gate = base_gate * (0.30 + 0.70 * transition_sync.clamp(0.0, 1.0))
        gate = gate * (1.0 - 0.35 * energy.unsqueeze(-1)) + trans_gate * (0.35 * energy.unsqueeze(-1))
        if alias_gate is not None:
            gate = gate * (0.25 + 0.75 * alias_gate.mean(dim=-1, keepdim=True).clamp(0.0, 1.0))
        h = self.mcg_norm(h + gate * msg)

        if "no_stconv" not in self.mode:
            st_msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            st_gate = torch.sigmoid(self.mcg_st_gate(torch.cat([stat, h, st_msg], dim=-1)))
            st_gate = st_gate * (1.0 - 0.50 * energy.unsqueeze(-1))
            if "closed_st" in self.mode:
                st_gate = 0.5 * st_gate
            h = self.mcg_st_norm(h + st_gate * st_msg)
        return h
