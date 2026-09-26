import torch
from torch import Tensor

from models.KMGONet import Model as KMGOModel
from utils.ExpConfigs import ExpConfigs


class Model(KMGOModel):
    """Mechanism-State Routed Graph Operator Network.

    MSRGO-Net keeps KMGO-Net's kernel-mechanism graph backbone but makes the
    adaptive-kernel bank deployment-conditioned: adaptive kernels are admitted
    only when the observed patch state is sparse enough, smooth enough, and
    old enough to benefit from a wider temporal support. Dense or highly active
    patches fall back to the static kernel bank plus mechanism graph.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)

    def _adaptive_need_logit(self, fixed_stat: Tensor) -> Tensor:
        density = fixed_stat[..., 0].clamp(0.0, 1.0)
        activity = fixed_stat[..., 3].clamp_min(0.0)
        trend = fixed_stat[..., 6].abs()
        span = fixed_stat[..., 9].clamp(0.0, 1.0)
        recency = fixed_stat[..., 10].clamp(0.0, 1.0)

        sparse = (1.0 - density).clamp(0.0, 1.0)
        smooth = (1.0 / (1.0 + 4.0 * activity + 0.05 * trend)).clamp(0.0, 1.0)
        stale = (0.5 + 0.5 * recency).clamp(0.5, 1.0)
        compact = (1.0 - 0.5 * span).clamp(0.5, 1.0)
        need = (sparse * smooth * stale * compact).clamp(0.02, 0.98)
        if "soft_need" in self.mode:
            need = (0.5 * need + 0.25).clamp(0.02, 0.98)
        elif "hard_need" in self.mode:
            need = (need > 0.20).to(dtype=fixed_stat.dtype) * 0.96 + (need <= 0.20).to(dtype=fixed_stat.dtype) * 0.04
        return torch.logit(need)

    def _adaptive_bank_bias(self, fixed_stat: Tensor, n_scales: int) -> Tensor:
        learned = super()._adaptive_bank_bias(fixed_stat, n_scales)
        if "no_need_router" in self.mode:
            return learned
        route = self._adaptive_need_logit(fixed_stat).unsqueeze(2).expand(-1, -1, n_scales, -1)
        strength = 1.0
        if "weak_need" in self.mode:
            strength = 0.5
        elif "strong_need" in self.mode:
            strength = 1.5
        return learned + strength * route
