import torch
import torch.nn as nn
from torch import Tensor

from models.AMPGNet import MLP
from models.CAGNet import Model as CAGModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGModel):
    """Mechanism-State Canonicalization and Admission Network.

    MSCANet upgrades CAGNet from scale admission to a mechanism-canonical
    patch backbone.  Before admission, the multiresolution coordinate is
    transported through a small bank of deployment-visible mechanism-state
    prototypes.  The transport is closed-biased and level-preserving: it can
    recover CAGNet when the learned transport is unused, but it can also map
    value-similar patches observed under different reliability/recency/co-
    observation states into a shared canonical forecasting coordinate.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.n_canon = int(getattr(configs, "n_canon_states", 4) or 4)
        route_stat_dim = 36
        route_token_dim = route_stat_dim + 3 * self.d_model
        self.canon_router = MLP(route_stat_dim, self.hidden, self.n_canon, self.dropout)
        self.canon_token_bank = nn.Parameter(torch.randn(self.n_canon, self.d_model) * 0.01)
        self.canon_token_delta = MLP(route_token_dim, self.hidden, self.d_model, self.dropout)
        self.canon_token_gate = MLP(route_token_dim, self.hidden, self.d_model, self.dropout)
        self.canon_stat_delta = MLP(route_stat_dim, self.hidden, 12, self.dropout)
        self.canon_token_norm = nn.LayerNorm(self.d_model)
        self.canon_stat_norm = nn.LayerNorm(12)
        self._latest_canon_gate: Tensor | None = None
        self._latest_canon_entropy: Tensor | None = None

        # Closed-by-default transport.  The model must learn evidence for
        # canonical movement instead of perturbing every multiresolution token.
        last = self.canon_token_gate.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, -2.0)
        stat_last = self.canon_stat_delta.net[-1]
        if isinstance(stat_last, nn.Linear):
            nn.init.zeros_(stat_last.weight)
            nn.init.zeros_(stat_last.bias)

    def _canonicalize(
        self,
        fixed_token: Tensor,
        fixed_stat: Tensor,
        scale_token: Tensor,
        scale_stat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if "no_canon" in self.mode or "no_canonical" in self.mode:
            zero_gate = torch.zeros_like(scale_token)
            zero_entropy = torch.zeros_like(scale_stat[..., 0])
            return scale_token, scale_stat, zero_gate, zero_entropy

        route_stat = torch.cat([fixed_stat, scale_stat, (fixed_stat - scale_stat).abs()], dim=-1)
        route_token = torch.cat(
            [route_stat, fixed_token, scale_token, (fixed_token - scale_token).abs()],
            dim=-1,
        )
        proto_weight = torch.softmax(self.canon_router(route_stat), dim=-1)
        proto_shift = torch.einsum("...k,kd->...d", proto_weight, self.canon_token_bank)
        raw_delta = torch.tanh(self.canon_token_delta(route_token) + proto_shift)
        gate = torch.sigmoid(self.canon_token_gate(route_token))
        if "canon_open" in self.mode:
            gate = 0.5 + 0.5 * gate
        elif "canon_tiny" in self.mode or "canon_closed" in self.mode:
            gate = 0.5 * gate
        canon_token = self.canon_token_norm(scale_token + gate * raw_delta)

        # Transport mechanism dimensions and rough dynamics, while preserving
        # the level coordinates that make the patch forecast valid.
        stat_gate = gate.mean(dim=-1, keepdim=True)
        stat_delta = 0.10 * torch.tanh(self.canon_stat_delta(route_stat))
        stat_mask = torch.zeros(12, device=scale_stat.device, dtype=scale_stat.dtype)
        # density, obs, variance, slope, first_t, last_t, span, recency, center
        stat_mask[[0, 1, 3, 6, 7, 8, 9, 10, 11]] = 1.0
        canon_stat = scale_stat + stat_gate * stat_delta * stat_mask.view(1, 1, 1, 12)
        canon_stat = torch.nan_to_num(canon_stat, nan=0.0, posinf=0.0, neginf=0.0)
        lower = torch.full((12,), -1.0e6, device=scale_stat.device, dtype=scale_stat.dtype)
        upper = torch.full((12,), 1.0e6, device=scale_stat.device, dtype=scale_stat.dtype)
        bounded = torch.tensor([0, 1, 7, 8, 9, 10, 11], device=scale_stat.device)
        lower = lower.scatter(0, bounded, 0.0)
        upper = upper.scatter(0, bounded, 1.0)
        canon_stat = torch.minimum(torch.maximum(canon_stat, lower.view(1, 1, 1, 12)), upper.view(1, 1, 1, 12))
        # Keep exact value-level coordinates level-preserving without in-place
        # writes on tensors that participate in autograd.
        preserve = torch.zeros(12, device=scale_stat.device, dtype=scale_stat.dtype)
        preserve = preserve.scatter(0, torch.tensor([2, 4, 5], device=scale_stat.device), 1.0)
        canon_stat = canon_stat * (1.0 - preserve.view(1, 1, 1, 12)) + scale_stat * preserve.view(1, 1, 1, 12)

        entropy = -(proto_weight * proto_weight.clamp_min(1e-8).log()).sum(dim=-1)
        return canon_token, canon_stat, gate, entropy

    def _admit(
        self,
        fixed_token: Tensor,
        fixed_stat: Tensor,
        scale_token: Tensor,
        scale_stat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scale_token, scale_stat, canon_gate, canon_entropy = self._canonicalize(
            fixed_token, fixed_stat, scale_token, scale_stat
        )
        self._latest_canon_gate = canon_gate
        self._latest_canon_entropy = canon_entropy
        return super()._admit(fixed_token, fixed_stat, scale_token, scale_stat)

    def forward(self, *args, **kwargs) -> dict:
        out = super().forward(*args, **kwargs)
        if self._latest_canon_gate is not None:
            self.latest_diag.update(
                {
                    "canon_gate_mean": float(self._latest_canon_gate.mean().detach().cpu()),
                    "canon_entropy": float(self._latest_canon_entropy.mean().detach().cpu())
                    if self._latest_canon_entropy is not None
                    else 0.0,
                }
            )
        return out
