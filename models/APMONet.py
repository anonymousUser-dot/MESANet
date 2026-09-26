import torch
import torch.nn as nn
from torch import Tensor

from models.AMPGNet import MLP
from models.CAGNet import Model as CAGModel
from utils.ExpConfigs import ExpConfigs


class Model(CAGModel):
    """Amplitude-Preserving Mechanism Operator Network.

    APMO-Net keeps CAGNet's level and admitted multiresolution patch statistics
    unchanged.  The only new operation is a small closed feature-space
    modulation conditioned on deployment-visible mechanism state.  This tests
    the hypothesis suggested by the v493-v497 evidence: wearable forecasting
    gains should preserve the amplitude coordinate and use mechanism state only
    to adjust representation geometry, not to rewrite the value anchor itself.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.apmo_delta = MLP(12 + self.d_model, self.hidden, self.d_model, self.dropout)
        self.apmo_gate = MLP(12 + self.d_model, self.hidden, 1, self.dropout)
        self.apmo_norm = nn.LayerNorm(self.d_model)
        last = self.apmo_gate.net[-1]
        if isinstance(last, nn.Linear):
            bias = -2.0
            if "open_apmo" in self.mode:
                bias = -0.5
            elif "very_closed_apmo" in self.mode:
                bias = -3.0
            nn.init.constant_(last.bias, bias)

    def _apmo_bound(self) -> float:
        if "weak_apmo" in self.mode:
            return 0.05
        if "strong_apmo" in self.mode:
            return 0.20
        if "very_strong_apmo" in self.mode:
            return 0.35
        return 0.10

    def _apmo_guard(self, stat: Tensor) -> Tensor:
        density = stat[..., 0:1].clamp(0.0, 1.0)
        var = stat[..., 3:4].clamp_min(0.0)
        slope = stat[..., 6:7].abs()
        if "sparse_apmo" in self.mode:
            return (1.0 - density).clamp(0.0, 1.0)
        if "active_apmo" in self.mode:
            return (var / (var + 0.05)).clamp(0.0, 1.0)
        if "transition_apmo" in self.mode:
            return (slope / (slope + 0.10)).clamp(0.0, 1.0)
        if "state_guard_apmo" in self.mode:
            sparse = (1.0 - density).clamp(0.0, 1.0)
            active = (var / (var + 0.05)).clamp(0.0, 1.0)
            transition = (slope / (slope + 0.10)).clamp(0.0, 1.0)
            return (0.34 * sparse + 0.33 * active + 0.33 * transition).clamp(0.0, 1.0)
        return torch.ones_like(density)

    def _admit(
        self,
        fixed_token: Tensor,
        fixed_stat: Tensor,
        scale_token: Tensor,
        scale_stat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        token, stat, scalar = super()._admit(fixed_token, fixed_stat, scale_token, scale_stat)
        if "no_apmo" in self.mode:
            return token, stat, scalar
        feat = torch.cat([stat, token], dim=-1)
        gate = torch.sigmoid(self.apmo_gate(feat))
        if "mean_apmo" in self.mode:
            gate = torch.full_like(gate, 0.5)
        delta = torch.tanh(self.apmo_delta(feat))
        guard = self._apmo_guard(stat)
        token = self.apmo_norm(token + self._apmo_bound() * gate * guard * delta)
        return token, stat, scalar
