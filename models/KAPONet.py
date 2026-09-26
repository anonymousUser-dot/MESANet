import torch
import torch.nn as nn
from torch import Tensor

from models.AMPGNet import MLP
from models.CTPGNet import Model as CTPGModel
from models.MAPONet import Model as MAPOModel
from utils.ExpConfigs import ExpConfigs


class Model(MAPOModel):
    """Kernel-Admitted Patch Operator Network.

    KAPO-Net keeps CAGNet's fixed continuous-time patch kernels as a native
    candidate bank and appends mechanism-adaptive kernels as extra candidates.
    The scale selector must therefore choose adaptive kernels against the
    original fixed kernels instead of replacing the whole scale branch. This
    gives the backbone an explicit fallback path when the mechanism-state
    estimate is weak.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.kapo_bank_norm = nn.LayerNorm(12)
        self.kapo_adaptive_prior = MLP(12, self.hidden, 1, self.dropout)
        last = self.kapo_adaptive_prior.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            bias = -1.0
            mode = str(getattr(configs, "ablation_name", "") or "").lower()
            if "open_bank" in mode:
                bias = 0.0
            elif "very_closed_bank" in mode:
                bias = -2.5
            elif "weak_closed_bank" in mode:
                bias = -0.5
            nn.init.constant_(last.bias, bias)

    def _adaptive_bank_bias(self, fixed_stat: Tensor, n_scales: int) -> Tensor:
        bias = self.kapo_adaptive_prior(fixed_stat).permute(0, 1, 3, 2)
        return bias.expand(-1, -1, n_scales, -1)

    def _scale_branch(self, x: Tensor, x_mask: Tensor, t: Tensor, fixed_stat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        static_mid, center_t, static_has_obs = CTPGModel._patch_stats(self, x, x_mask, t)
        if self._ct_scale_stat is None:
            raise RuntimeError("KAPONet static CTPG stats missing.")
        static_stat = self._ct_scale_stat
        adaptive_stat, adaptive_center_t, adaptive_has_obs = self._adaptive_scale_stats(x, x_mask, t, fixed_stat)

        n_patch = min(static_stat.size(3), adaptive_stat.size(3), fixed_stat.size(2))
        static_stat = static_stat[:, :, :, :n_patch, :]
        adaptive_stat = adaptive_stat[:, :, :, :n_patch, :]
        center_t = center_t[:, :, :n_patch, :]
        fixed_stat = fixed_stat[:, :, :n_patch, :]
        static_has_obs = static_has_obs[:, :, :n_patch]
        adaptive_has_obs = adaptive_has_obs[:, :, :n_patch]

        if "kapo_static_only" in self.mode or "no_adaptive_bank" in self.mode:
            scale_stat = static_stat
            has_obs = static_has_obs
            adaptive_start = scale_stat.size(2)
        elif "kapo_adaptive_only" in self.mode or "adaptive_only_bank" in self.mode:
            scale_stat = adaptive_stat
            has_obs = adaptive_has_obs
            adaptive_start = 0
        else:
            scale_stat = torch.cat([static_stat, adaptive_stat], dim=2)
            has_obs = torch.maximum(static_has_obs, adaptive_has_obs)
            adaptive_start = static_stat.size(2)

        if "bank_norm" in self.mode:
            scale_stat = self.kapo_bank_norm(scale_stat)
        batch, n_vars, n_candidates, n_patch, _ = scale_stat.shape
        var = self.var_emb.to(scale_stat.device, scale_stat.dtype).view(1, n_vars, 1, 1, self.d_model)
        var = var.expand(batch, -1, n_candidates, n_patch, -1)
        te = self.time_enc(center_t).unsqueeze(2).expand(-1, -1, n_candidates, -1, -1)
        base = torch.cat([scale_stat, te, var], dim=-1)
        token_k = self.ct_scale_op(base)
        logits = self.ct_scale_score(base).squeeze(-1)

        if "no_state_scale" not in self.mode:
            state_stat = self._value_only_stat(fixed_stat) if "value_only_state" in self.mode else fixed_stat
            state = self.state_emb(state_stat)
            bias = self.state_scale_bias(torch.cat([state_stat, state], dim=-1)).permute(0, 1, 3, 2)
            repeats = max(1, (n_candidates + bias.size(2) - 1) // bias.size(2))
            logits = logits + bias.repeat(1, 1, repeats, 1)[:, :, :n_candidates, :]

        if "no_mapo_score" not in self.mode:
            fixed_expand = fixed_stat.unsqueeze(2).expand_as(scale_stat)
            logits = logits + self.mapo_score_bias(torch.cat([fixed_expand, scale_stat], dim=-1)).squeeze(-1)

        if adaptive_start < n_candidates and "no_adaptive_prior" not in self.mode:
            logits[:, :, adaptive_start:, :] = logits[:, :, adaptive_start:, :] + self._adaptive_bank_bias(
                fixed_stat, n_candidates - adaptive_start
            )

        if "single_scale" in self.mode:
            idx = min(1, n_candidates - 1)
            alpha = torch.zeros(batch, n_vars, n_candidates, n_patch, 1, device=scale_stat.device, dtype=scale_stat.dtype)
            alpha[:, :, idx, :, :] = 1.0
        elif "uniform_scale" in self.mode:
            alpha = torch.full(
                (batch, n_vars, n_candidates, n_patch, 1),
                1.0 / n_candidates,
                device=scale_stat.device,
                dtype=scale_stat.dtype,
            )
        else:
            alpha = torch.softmax(logits, dim=2).unsqueeze(-1)

        token = (alpha * token_k).sum(dim=2)
        fused_stat = (alpha * scale_stat).sum(dim=2)
        token = self.ct_norm(token + self.var_emb.to(token.device, token.dtype).view(1, n_vars, 1, self.d_model))

        with torch.no_grad():
            if adaptive_start < n_candidates:
                self.latest_kapo_adaptive_mass = alpha[:, :, adaptive_start:, :, :].sum(dim=2).mean().detach()
            else:
                self.latest_kapo_adaptive_mass = torch.zeros((), device=scale_stat.device, dtype=scale_stat.dtype)
            self.latest_mapo_width_factor = self._mechanism_width_factor(fixed_stat).detach()
            self._ct_scale_stat = scale_stat
        return token, fused_stat, has_obs
