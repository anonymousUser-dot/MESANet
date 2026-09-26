import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from models.CTPGNet import Model as CTPGModel
from utils.ExpConfigs import ExpConfigs


class Model(CTPGModel):
    """Closed-Admission Multiresolution Graph Network.

    The backbone retains a fixed-support value coordinate and constructs
    multiresolution coordinates directly from causal irregular observations.
    A closed-biased admission map uses deployment-visible support statistics to
    control how much scale evidence enters each patch token before aggregation.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.mode = str(getattr(configs, "ablation_name", "") or "").lower()
        mode_tokens = set(self.mode.split("_"))
        self.use_affine_anchor = "affine_anchor" in self.mode or "aa" in mode_tokens
        self.use_standard_revin = "standard_revin" in self.mode
        self.use_masked_revin = "masked_revin" in self.mode
        self.disable_anchor_restore = "no_anchor_restore" in self.mode
        if self.use_masked_revin:
            self.masked_revin_weight = nn.Parameter(torch.ones(1, 1, self.n_vars))
            self.masked_revin_bias = nn.Parameter(torch.zeros(1, 1, self.n_vars))
        else:
            self.register_parameter("masked_revin_weight", None)
            self.register_parameter("masked_revin_bias", None)
        self.use_horizon_simplex = "horizon_simplex" in self.mode or "hs" in mode_tokens
        self.use_subject_balancing = "subject_balanced" in self.mode or "subject_groupdro" in self.mode
        self.use_subject_groupdro = "subject_groupdro" in self.mode
        self.use_subject_context = "subject_context" in self.mode or "context_adapter" in self.mode
        self.use_pose_canonical = "pose_canonical" in self.mode or "pose_reliable" in self.mode
        self.use_pose_reliable = "pose_reliable" in self.mode
        self.use_ordered_patch = "ordered_patch" in self.mode
        self.use_ordered_patch_only = "ordered_patch_only" in self.mode
        self.use_closed_ordered_patch = "ordered_patch_closed" in self.mode
        self.use_hierarchical_ordered_patch = "ordered_patch_hierarchical" in self.mode
        self.use_separable_ordered_patch = "ordered_patch_separable" in self.mode
        self.use_risk_supervised_ordered_patch = "ordered_patch_risk_supervised" in self.mode
        self.use_orthogonal_ordered_patch = "ordered_patch_orthogonal" in self.mode
        self.use_horizon_patch_query = "horizon_patch_query" in self.mode
        self.use_waveform_token = "waveform_token" in self.mode
        self.use_waveform_residual = "waveform_token_residual" in self.mode
        self.use_parallel_waveform = "waveform_token_parallel" in self.mode
        self.use_waveform_crossattn = "waveform_crossattn" in self.mode
        self.use_level_shape_decomp = "level_shape_decomp" in self.mode
        self.use_dct_dual_coordinate = "dct_dual_coordinate" in self.mode
        self.dual_no_high = "dct_dual_no_high" in self.mode
        self.dual_fixed_level = "dct_dual_fixed_level" in self.mode
        self.dual_no_causal = "dct_dual_no_causal" in self.mode
        self.dual_no_orthogonal = "dct_dual_no_orthogonal" in self.mode
        self.dual_low_subspace_only = "dct_dual_low_subspace_only" in self.mode
        self.dual_soft_orthogonal = "dct_dual_soft_orthogonal" in self.mode
        self.dual_shared_transfer = "dct_dual_shared_transfer" in self.mode
        self.dual_random_basis = "dct_dual_random_basis" in self.mode
        self.dual_fixed_orth_weight: float | None = None
        if "dct_dual_rho_zero" in self.mode:
            self.dual_fixed_orth_weight = 0.0
        elif "dct_dual_rho_half" in self.mode:
            self.dual_fixed_orth_weight = 0.5
        elif "dct_dual_rho_one" in self.mode:
            self.dual_fixed_orth_weight = 1.0
        rho_override = float(getattr(configs, "mesa_dual_rho", -1.0))
        if rho_override >= 0.0:
            if rho_override > 1.0:
                raise ValueError("mesa_dual_rho must lie in [0, 1]")
            self.dual_fixed_orth_weight = rho_override
        self.use_lowrank_trajectory = "lowrank_trajectory" in self.mode
        self.use_lowrank_trajectory_only = "lowrank_trajectory_only" in self.mode
        self.use_spectral_trajectory = "spectral_trajectory" in self.mode
        self.use_spectral_global = "spectral_global" in self.mode
        self.use_spectral_support_only = "spectral_support_only" in self.mode
        self.pose_groups = self._wearable_pose_groups(str(getattr(configs, "dataset_name", "")))
        self._pose_frames: list[tuple[tuple[int, ...], Tensor]] = []
        self._pose_frame_usage: Tensor | None = None
        self.subject_dro_mix = 0.5
        self.subject_dro_temperature = 0.05
        self.subject_context_scale = 0.25
        if self.use_subject_context:
            self.subject_context = MLP(8, self.hidden, 2 * self.d_model, self.dropout)
            self.subject_context_norm = nn.LayerNorm(self.d_model)
            last_context = self.subject_context.net[-1]
            if isinstance(last_context, nn.Linear):
                nn.init.zeros_(last_context.weight)
                nn.init.zeros_(last_context.bias)
        else:
            self.subject_context = None
            self.subject_context_norm = None
        if self.use_ordered_patch:
            self.ordered_n_patch = int(math.ceil(float(configs.seq_len) / float(self.patch_len)))
            self.ordered_patch_norm = nn.LayerNorm(self.d_model)
            self.ordered_patch_head = nn.Sequential(
                nn.Linear(self.ordered_n_patch * self.d_model, self.hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, self.pred_len),
            )
            if self.use_separable_ordered_patch:
                self.ordered_patch_gate = None
            else:
                self.ordered_patch_gate = MLP(12 + self.d_model, self.hidden, self.pred_len, self.dropout)
                gate_last = self.ordered_patch_gate.net[-1]
                if isinstance(gate_last, nn.Linear):
                    if self.use_hierarchical_ordered_patch:
                        nn.init.zeros_(gate_last.weight)
                        nn.init.zeros_(gate_last.bias)
                    else:
                        nn.init.constant_(gate_last.bias, -2.5 if self.use_closed_ordered_patch else -1.0)
            if self.use_hierarchical_ordered_patch or self.use_separable_ordered_patch:
                self.ordered_global_logit = nn.Parameter(torch.tensor(-2.0))
            else:
                self.register_parameter("ordered_global_logit", None)
            if self.use_separable_ordered_patch:
                self.ordered_horizon_logit = nn.Parameter(torch.zeros(1, self.pred_len, 1))
                self.ordered_variable_logit = nn.Parameter(torch.zeros(1, 1, self.n_vars))
            else:
                self.register_parameter("ordered_horizon_logit", None)
                self.register_parameter("ordered_variable_logit", None)
        else:
            self.ordered_n_patch = 0
            self.ordered_patch_norm = None
            self.ordered_patch_head = None
            self.ordered_patch_gate = None
            self.register_parameter("ordered_global_logit", None)
            self.register_parameter("ordered_horizon_logit", None)
            self.register_parameter("ordered_variable_logit", None)
        if self.use_lowrank_trajectory:
            self.trajectory_n_patch = int(math.ceil(float(configs.seq_len) / float(self.patch_len)))
            self.trajectory_rank = min(
                max(1, int(getattr(configs, "mesa_trajectory_rank", 4))),
                self.trajectory_n_patch,
            )
            position = torch.arange(self.trajectory_n_patch, dtype=torch.float32).unsqueeze(1)
            frequency = torch.arange(self.trajectory_rank, dtype=torch.float32).unsqueeze(0)
            basis = torch.cos(
                math.pi
                * (position + 0.5)
                * frequency
                / float(self.trajectory_n_patch)
            )
            basis[:, 0] *= math.sqrt(1.0 / float(self.trajectory_n_patch))
            if self.trajectory_rank > 1:
                basis[:, 1:] *= math.sqrt(2.0 / float(self.trajectory_n_patch))
            self.register_buffer("trajectory_basis", basis, persistent=True)
            self.trajectory_norm = nn.LayerNorm(self.d_model)
            trajectory_in = (1 + self.trajectory_rank) * self.d_model
            self.trajectory_hidden = max(16, min(self.hidden, self.d_model))
            self.trajectory_head = nn.Sequential(
                nn.Linear(trajectory_in, self.trajectory_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.trajectory_hidden, self.pred_len),
            )
            self.trajectory_gate = MLP(
                12 + self.d_model,
                self.trajectory_hidden,
                self.pred_len,
                self.dropout,
            )
            trajectory_gate_last = self.trajectory_gate.net[-1]
            if isinstance(trajectory_gate_last, nn.Linear):
                nn.init.constant_(trajectory_gate_last.bias, -1.0)
        else:
            self.trajectory_n_patch = 0
            self.trajectory_rank = 0
            self.trajectory_hidden = 0
            self.register_buffer("trajectory_basis", None, persistent=False)
            self.trajectory_norm = None
            self.trajectory_head = None
            self.trajectory_gate = None
        if self.use_spectral_trajectory:
            self.spectral_n_patch = int(math.ceil(float(configs.seq_len) / float(self.patch_len)))
            position = torch.arange(self.spectral_n_patch, dtype=torch.float32).unsqueeze(1)
            frequency = torch.arange(self.spectral_n_patch, dtype=torch.float32).unsqueeze(0)
            spectral_basis = torch.cos(
                math.pi * (position + 0.5) * frequency / float(self.spectral_n_patch)
            )
            spectral_basis[:, 0] *= math.sqrt(1.0 / float(self.spectral_n_patch))
            if self.spectral_n_patch > 1:
                spectral_basis[:, 1:] *= math.sqrt(2.0 / float(self.spectral_n_patch))
            self.register_buffer("spectral_basis", spectral_basis, persistent=True)
            spectral_hidden = max(16, min(self.hidden, self.d_model))
            te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
            self.spectral_norm = nn.LayerNorm(self.d_model)
            self.spectral_gate = (
                None if self.use_spectral_global else MLP(14, spectral_hidden, 1, self.dropout)
            )
            self.spectral_frequency = nn.Linear(1, self.d_model)
            self.spectral_query = MLP(
                self.d_model + te_dim + 1,
                spectral_hidden,
                self.d_model,
                self.dropout,
            )
            self.spectral_key = nn.Linear(self.d_model, self.d_model, bias=False)
            self.spectral_value = nn.Linear(self.d_model, self.d_model, bias=False)
            self.spectral_context_norm = nn.LayerNorm(self.d_model)
            if self.use_spectral_global:
                frequency_prior = torch.linspace(2.0, -2.0, self.spectral_n_patch)
                self.spectral_global_logit = nn.Parameter(
                    frequency_prior.view(1, 1, self.spectral_n_patch, 1)
                )
            else:
                self.register_parameter("spectral_global_logit", None)
            if self.spectral_gate is not None:
                spectral_gate_last = self.spectral_gate.net[-1]
                if isinstance(spectral_gate_last, nn.Linear):
                    nn.init.constant_(spectral_gate_last.bias, -1.0)
        else:
            self.spectral_n_patch = 0
            self.register_buffer("spectral_basis", None, persistent=False)
            self.spectral_norm = None
            self.spectral_gate = None
            self.spectral_frequency = None
            self.spectral_query = None
            self.spectral_key = None
            self.spectral_value = None
            self.spectral_context_norm = None
            self.register_parameter("spectral_global_logit", None)
        if self.use_horizon_patch_query:
            te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
            self.hpq_query = MLP(self.d_model + te_dim + 1, self.hidden, self.d_model, self.dropout)
            self.hpq_key = nn.Linear(self.d_model, self.d_model, bias=False)
            self.hpq_value = nn.Linear(self.d_model, self.d_model, bias=False)
            self.hpq_gate = MLP(12 + 3 * self.d_model, self.hidden, self.d_model, self.dropout)
            self.hpq_norm = nn.LayerNorm(self.d_model)
            hpq_last = self.hpq_gate.net[-1]
            if isinstance(hpq_last, nn.Linear):
                nn.init.constant_(hpq_last.bias, -1.0)
        else:
            self.hpq_query = None
            self.hpq_key = None
            self.hpq_value = None
            self.hpq_gate = None
            self.hpq_norm = None
        if self.use_waveform_token:
            waveform_in = 3 * self.patch_len + self.d_model
            self.waveform_proj = MLP(waveform_in, self.hidden, self.d_model, self.dropout)
            self.waveform_gate = MLP(12 + 2 * self.d_model, self.hidden, self.d_model, self.dropout)
            self.waveform_norm = nn.LayerNorm(self.d_model)
            waveform_last = self.waveform_gate.net[-1]
            if isinstance(waveform_last, nn.Linear):
                nn.init.constant_(waveform_last.bias, -2.5 if self.use_waveform_residual else -1.0)
        else:
            self.waveform_proj = None
            self.waveform_gate = None
            self.waveform_norm = None
        if self.use_waveform_crossattn:
            heads = max(1, min(int(configs.n_heads), self.d_model // 16))
            while self.d_model % heads != 0:
                heads -= 1
            self.waveform_cross_attn = nn.MultiheadAttention(
                self.d_model,
                heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.waveform_cross_gate = MLP(
                12 + 2 * self.d_model,
                self.hidden,
                self.d_model,
                self.dropout,
            )
            self.waveform_cross_norm = nn.LayerNorm(self.d_model)
            cross_last = self.waveform_cross_gate.net[-1]
            if isinstance(cross_last, nn.Linear):
                nn.init.constant_(cross_last.bias, -2.0)
        else:
            self.waveform_cross_attn = None
            self.waveform_cross_gate = None
            self.waveform_cross_norm = None
        if self.use_level_shape_decomp:
            self.level_shape_n_patch = int(math.ceil(float(configs.seq_len) / float(self.patch_len)))
            heads = max(1, min(int(configs.n_heads), self.d_model // 16))
            self.level_var_attn = nn.MultiheadAttention(
                self.d_model,
                heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.shape_var_attn = nn.MultiheadAttention(
                self.d_model,
                heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.level_var_norm = nn.LayerNorm(self.d_model)
            self.shape_var_norm = nn.LayerNorm(self.d_model)
            self.level_shape_level_head = MLP(
                self.d_model + 12,
                self.hidden,
                1,
                self.dropout,
            )
            self.level_shape_shape_head = nn.Sequential(
                nn.Linear(self.level_shape_n_patch * self.d_model, self.hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, self.pred_len),
            )
        else:
            self.level_shape_n_patch = 0
            self.level_var_attn = None
            self.shape_var_attn = None
            self.level_var_norm = None
            self.shape_var_norm = None
            self.level_shape_level_head = None
            self.level_shape_shape_head = None
        if self.use_dct_dual_coordinate:
            self.dual_n_patch = int(math.ceil(float(configs.seq_len) / float(self.patch_len)))
            self.dual_rank = min(4, self.pred_len)
            if self.dual_random_basis:
                generator = torch.Generator().manual_seed(2027 + self.pred_len)
                basis, _ = torch.linalg.qr(
                    torch.randn(
                        self.pred_len,
                        self.dual_rank,
                        generator=generator,
                        dtype=torch.float32,
                    ),
                    mode="reduced",
                )
            else:
                horizon = torch.arange(self.pred_len, dtype=torch.float32).unsqueeze(1)
                frequency = torch.arange(self.dual_rank, dtype=torch.float32).unsqueeze(0)
                basis = torch.cos(
                    math.pi
                    * (horizon + 0.5)
                    * frequency
                    / float(self.pred_len)
                )
                basis[:, 0] *= math.sqrt(1.0 / float(self.pred_len))
                if self.dual_rank > 1:
                    basis[:, 1:] *= math.sqrt(2.0 / float(self.pred_len))
            self.register_buffer("dual_dct_basis", basis, persistent=True)
            self.dual_level_conv = nn.Conv1d(
                self.d_model,
                self.d_model,
                kernel_size=3,
                padding=2,
                groups=self.d_model,
            )
            self.dual_shape_conv = nn.Conv1d(
                self.d_model,
                self.d_model,
                kernel_size=3,
                padding=4,
                dilation=2,
                groups=self.d_model,
            )
            self.dual_level_mix = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
            self.dual_shape_mix = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
            heads = max(1, min(int(configs.n_heads), self.d_model // 16))
            while self.d_model % heads != 0:
                heads -= 1
            self.dual_level_var = nn.MultiheadAttention(
                self.d_model, heads, dropout=self.dropout, batch_first=True
            )
            if self.dual_shared_transfer:
                self.dual_shape_var = self.dual_level_var
                # Two coordinate-local channel mixers exactly replace the
                # parameters removed by sharing one d_model-wide MHA block.
                self.dual_level_capacity = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                self.dual_shape_capacity = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.d_model),
                )
            else:
                self.dual_shape_var = nn.MultiheadAttention(
                    self.d_model, heads, dropout=self.dropout, batch_first=True
                )
                self.dual_level_capacity = None
                self.dual_shape_capacity = None
            self.dual_level_norm = nn.LayerNorm(self.d_model)
            self.dual_shape_norm = nn.LayerNorm(self.d_model)
            self.dual_level_head = nn.Sequential(
                nn.Linear(self.dual_n_patch * self.d_model, self.hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, self.dual_rank),
            )
            self.dual_shape_head = nn.Sequential(
                nn.Linear(self.dual_n_patch * self.d_model, self.hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, self.pred_len),
            )
            if self.dual_soft_orthogonal and self.dual_fixed_orth_weight is None:
                self.dual_orth_logit = nn.Parameter(torch.tensor(0.0))
            else:
                self.register_parameter("dual_orth_logit", None)
        else:
            self.dual_n_patch = 0
            self.dual_rank = 0
            self.register_buffer("dual_dct_basis", None, persistent=False)
            self.dual_level_conv = None
            self.dual_shape_conv = None
            self.dual_level_mix = None
            self.dual_shape_mix = None
            self.dual_level_var = None
            self.dual_shape_var = None
            self.dual_level_capacity = None
            self.dual_shape_capacity = None
            self.dual_level_norm = None
            self.dual_shape_norm = None
            self.dual_level_head = None
            self.dual_shape_head = None
            self.register_parameter("dual_orth_logit", None)
        self.state_emb = MLP(12, self.hidden, self.d_model, self.dropout)
        self.state_scale_bias = MLP(12 + self.d_model, self.hidden, self.n_scales, self.dropout)
        route_in = 36 + 3 * self.d_model
        self.admit_scalar = MLP(route_in, self.hidden, 1, self.dropout)
        self.admit_vector = MLP(route_in, self.hidden, self.d_model, self.dropout)
        self.admit_norm = nn.LayerNorm(self.d_model)
        concat_in = (1 + self.n_scales) * self.d_model
        concat_hidden = max(
            1,
            int(
                round(
                    (
                        self._mlp_param_count(route_in, self.hidden, 1)
                        + self._mlp_param_count(route_in, self.hidden, self.d_model)
                        - self.d_model
                    )
                    / float(concat_in + self.d_model + 1)
                )
            ),
        )
        self.ms_concat_proj = MLP(concat_in, concat_hidden, self.d_model, self.dropout)
        self.global_admit_logit = nn.Parameter(torch.zeros(1))

        # Horizon-simplex modules belong only to that named experimental mode;
        # the released MESANet does not carry inactive parameters.
        if self.use_horizon_simplex:
            te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
            self.horizon_scale_pool = nn.Linear(self.d_model + 12, 1)
            self.horizon_query = MLP(te_dim + 1, self.hidden, self.d_model, self.dropout)
            self.horizon_scale_key = nn.Linear(self.d_model, self.d_model, bias=False)
            self.horizon_state_bias = MLP(12 + te_dim + 1, self.hidden, self.n_scales + 1, self.dropout)
            self.horizon_simplex_norm = nn.LayerNorm(self.d_model)
            self.horizon_reliability_logit = nn.Parameter(torch.tensor(-2.0))
        else:
            self.horizon_scale_pool = None
            self.horizon_query = None
            self.horizon_scale_key = None
            self.horizon_state_bias = None
            self.horizon_simplex_norm = None
            self.register_parameter("horizon_reliability_logit", None)
        self._cag_fixed_stat: Tensor | None = None
        self._time_irregularity: Tensor | None = None
        self._cag_scale_tokens: Tensor | None = None
        self._cag_scale_alpha: Tensor | None = None
        last = self.admit_scalar.net[-1]
        if isinstance(last, nn.Linear):
            bias = -2.0
            if "open_admit" in self.mode:
                bias = 0.0
            elif "very_closed" in self.mode:
                bias = -3.0
            nn.init.constant_(last.bias, bias)

    @staticmethod
    def _mlp_param_count(in_dim: int, hidden: int, out_dim: int) -> int:
        return in_dim * hidden + hidden + hidden * out_dim + out_dim

    @staticmethod
    def _support_state_cols(device: torch.device) -> Tensor:
        # Q columns: density, obs, mean, var, first, last, slope,
        # first_t, last_t, span, recency, center.  The listed columns are
        # deployment-visible observation-support cues rather than values.
        return torch.tensor([0, 1, 7, 8, 9, 10], device=device)

    def _remove_support_state(self, stat: Tensor) -> Tensor:
        out = stat.clone()
        cols = self._support_state_cols(stat.device)
        out[..., cols] = 0.0
        return out

    def _shuffle_support_state(self, stat: Tensor) -> Tensor:
        out = stat.clone()
        if out.size(0) > 1:
            cols = self._support_state_cols(stat.device)
            perm = torch.randperm(out.size(0), device=stat.device)
            out[..., cols] = out[perm][..., cols]
        return out

    def _prepare_support_stat(self, stat: Tensor) -> Tensor:
        if "no_obs_state" in self.mode:
            return self._remove_support_state(stat)
        if "shuffle_state" in self.mode:
            return self._shuffle_support_state(stat)
        return stat

    def _fixed_branch(self, x: Tensor, x_mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        stat, center_t, has_obs = AMPGModel._patch_stats(self, x, x_mask, t)
        stat_used = self._prepare_support_stat(stat)
        token, _, _, alias_gate, _ = AMPGModel._make_tokens(self, stat_used, center_t)
        return token, stat_used, has_obs, alias_gate

    def _scale_branch(self, x: Tensor, x_mask: Tensor, t: Tensor, fixed_stat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stat, center_t, has_obs = CTPGModel._patch_stats(self, x, x_mask, t)
        if self._ct_scale_stat is None:
            raise RuntimeError("CAGNet CTPG stats missing.")
        scale_stat = self._prepare_support_stat(self._ct_scale_stat)
        batch, n_vars, n_scales, n_patch, _ = scale_stat.shape
        fixed_stat = fixed_stat[:, :, :n_patch, :]
        if "fixed_support_param_matched" in self.mode:
            scale_stat = fixed_stat.unsqueeze(2).expand(-1, -1, n_scales, -1, -1)
            center_t = fixed_stat[..., 11:12]
            has_obs = fixed_stat[..., 1]
            self._ct_scale_stat = scale_stat
        var = self.var_emb.to(scale_stat.device, scale_stat.dtype).view(1, n_vars, 1, 1, self.d_model)
        var = var.expand(batch, -1, n_scales, n_patch, -1)
        te = self.time_enc(center_t).unsqueeze(2).expand(-1, -1, n_scales, -1, -1)
        base = torch.cat([scale_stat, te, var], dim=-1)
        token_k = self.ct_scale_op(base)
        logits = self.ct_scale_score(base).squeeze(-1)
        if "no_state_scale" not in self.mode:
            state_stat = self._value_only_stat(fixed_stat) if "value_only_state" in self.mode else fixed_stat
            state = self.state_emb(state_stat)
            bias = self.state_scale_bias(torch.cat([state_stat, state], dim=-1)).permute(0, 1, 3, 2)
            logits = logits + bias
        if "landau_edge_heat" in self.mode:
            density = fixed_stat[..., 0].clamp(0.05, 1.0)
            activity = fixed_stat[..., 3].clamp_min(0.0)
            edge_stop = (1.0 / (1.0 + 4.0 * activity)).clamp(0.0, 1.0)
            target_scale = 1.0 + (density.rsqrt().clamp(1.0, 4.0) - 1.0) * edge_stop
            target_log_width = target_scale.log().unsqueeze(2)
            scale_log_width = self.ct_log_width.to(device=scale_stat.device, dtype=scale_stat.dtype).view(1, 1, n_scales, 1)
            logits = logits - 0.5 * (scale_log_width - target_log_width).square()
        elif "landau_heat" in self.mode or "landau_prior" in self.mode:
            density = fixed_stat[..., 0].clamp(0.05, 1.0)
            target_log_width = density.rsqrt().log().unsqueeze(2)
            scale_log_width = self.ct_log_width.to(device=scale_stat.device, dtype=scale_stat.dtype).view(1, 1, n_scales, 1)
            logits = logits - 0.5 * (scale_log_width - target_log_width).square()
        alpha = torch.softmax(logits, dim=2).unsqueeze(-1)
        self._cag_scale_tokens = token_k
        self._cag_scale_alpha = alpha.squeeze(-1)
        token = (alpha * token_k).sum(dim=2)
        fused_stat = (alpha * scale_stat).sum(dim=2)
        token = self.ct_norm(token + self.var_emb.to(token.device, token.dtype).view(1, n_vars, 1, self.d_model))
        return token, fused_stat, has_obs

    @staticmethod
    def _value_only_stat(stat: Tensor) -> Tensor:
        """Keep value-level entries and remove observation-support cues."""
        mask = torch.zeros(stat.size(-1), device=stat.device, dtype=stat.dtype)
        # Q columns: density, obs, mean, var, first, last, slope,
        # first_t, last_t, span, recency, center.
        keep = torch.tensor([2, 3, 4, 5, 6], device=stat.device)
        mask.scatter_(0, keep, 1.0)
        return stat * mask

    def _admit(self, fixed_token: Tensor, fixed_stat: Tensor, scale_token: Tensor, scale_stat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        route_fixed_stat = self._value_only_stat(fixed_stat) if "value_only_admit" in self.mode or "no_obs_state" in self.mode else fixed_stat
        route_scale_stat = self._value_only_stat(scale_stat) if "value_only_admit" in self.mode or "no_obs_state" in self.mode else scale_stat
        route = torch.cat(
            [
                route_fixed_stat,
                route_scale_stat,
                (route_fixed_stat - route_scale_stat).abs(),
                fixed_token,
                scale_token,
                (fixed_token - scale_token).abs(),
            ],
            dim=-1,
        )
        temperature = max(1e-4, float(getattr(self.configs, "cag_admit_temperature", 1.0)))
        scalar = torch.sigmoid(self.admit_scalar(route) / temperature)
        scalar_template = scalar
        if "random_admit" in self.mode:
            scalar = torch.rand_like(scalar)
            vector = scalar.expand_as(fixed_token)
        elif "mean_admit" in self.mode:
            scalar = torch.full_like(scalar, 0.5)
            vector = scalar.expand_as(fixed_token)
        elif "always_open_gate" in self.mode or "always_open_admit" in self.mode:
            scalar = torch.ones_like(scalar)
            vector = torch.ones_like(fixed_token)
        elif "global_trainable_gate" in self.mode or "global_admit" in self.mode:
            scalar = torch.sigmoid(self.global_admit_logit).to(dtype=scalar.dtype, device=scalar.device).view(1, 1, 1, 1)
            scalar = scalar.expand_as(scalar_template)
            vector = scalar.expand_as(fixed_token)
        elif "gap_scale_gate" in self.mode and self._time_irregularity is not None:
            gate = torch.sigmoid(30.0 * (self._time_irregularity.to(scalar.dtype) - 0.10)).view(-1, 1, 1, 1)
            scalar = gate.expand_as(scalar)
            vector = gate.expand_as(fixed_token)
        elif "scalar_admit" in self.mode or "very_closed" in self.mode or "open_admit" in self.mode:
            vector = scalar.expand_as(fixed_token)
        else:
            vector = torch.sigmoid(self.admit_vector(route) / temperature)
            vector = 0.5 * vector + 0.5 * scalar
        if "fixed_only" in self.mode:
            scalar = torch.zeros_like(scalar)
            vector = torch.zeros_like(vector)
        elif "scale_only" in self.mode:
            scalar = torch.ones_like(scalar)
            vector = torch.ones_like(vector)
        elif "landau_edge_heat_guarded" in self.mode:
            density = fixed_stat[..., 0:1].clamp(0.0, 1.0)
            activity = fixed_stat[..., 3:4].clamp_min(0.0)
            trend = fixed_stat[..., 6:7].abs()
            edge_stop = (1.0 / (1.0 + 4.0 * activity + 0.05 * trend)).clamp(0.0, 1.0)
            sparse_need = (1.0 - density).clamp(0.0, 1.0)
            guard = (sparse_need * edge_stop).clamp(0.0, 1.0)
            scalar = scalar * guard
            vector = vector * guard
        token = fixed_token + vector * (scale_token - fixed_token)
        stat = fixed_stat + scalar * (scale_stat - fixed_stat)
        return self.admit_norm(token), stat, scalar

    def _multiscale_concat(
        self,
        fixed_token: Tensor,
        fixed_stat: Tensor,
        scale_stat: Tensor,
        n_patch: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self._cag_scale_tokens is None:
            raise RuntimeError("CAGNet multiscale-concat control requires scale tokens.")
        token_k = self._cag_scale_tokens[:, :, :, :n_patch, :]
        fixed = fixed_token[:, :, :n_patch, :].unsqueeze(2)
        concat = torch.cat([fixed, token_k], dim=2).permute(0, 1, 3, 2, 4).flatten(-2)
        token = self.admit_norm(self.ms_concat_proj(concat))
        admit = torch.ones_like(fixed_stat[:, :, :n_patch, 0:1])
        return token, scale_stat[:, :, :n_patch, :], admit

    def _temporal_stage(self, token: Tensor) -> Tensor:
        """Apply exactly the temporal stage used by the default dependency operator."""
        batch, n_vars, n_patch, dim = token.shape
        pos = self.temporal_pos[:, :n_patch, :].to(token.device, token.dtype)
        return self.temporal(token.reshape(batch * n_vars, n_patch, dim) + pos).view(
            batch, n_vars, n_patch, dim
        )

    def _variable_stage(self, h: Tensor, stat: Tensor) -> Tensor:
        """Apply exactly the variable-transfer stage used after temporal attention."""
        if "no_graph" in self.mode:
            return h
        q = self.graph_q(h).permute(0, 2, 1, 3)
        k = self.graph_k(h).permute(0, 2, 1, 3)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1)), dim=-1)
        msg = torch.einsum("bpij,bjpd->bipd", attn, h)
        msg = self.graph_msg(msg)
        graph_stat = torch.zeros_like(stat) if "no_q_graph" in self.mode else stat
        gate = torch.sigmoid(self.graph_gate(torch.cat([h, msg, graph_stat], dim=-1)))
        return self.graph_norm(h + gate * msg)

    def _decoder_position_tail(
        self,
        h: Tensor,
        stat: Tensor,
        history_context: Tensor | None,
    ) -> Tensor:
        """Apply the shared post-graph blocks before decoder-position fusion."""
        if "no_stconv" not in self.mode:
            msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            gate = torch.sigmoid(self.ct_st_gate(torch.cat([stat, h], dim=-1)))
            h = self.ct_st_norm(h + gate * msg)
        if history_context is not None:
            context_params = self.subject_context(history_context)
            gamma, beta = context_params.chunk(2, dim=-1)
            h = self.subject_context_norm(
                (1.0 + self.subject_context_scale * torch.tanh(gamma)[:, None, None, :]) * h
                + self.subject_context_scale * beta[:, None, None, :]
            )
        return h

    def _basic_decoder_input(
        self,
        h: Tensor,
        stat: Tensor,
        has_obs: Tensor,
        y_t: Tensor,
        t: Tensor,
        horizon: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Construct the shared direct-decoder input from one dependency stream."""
        batch = h.size(0)
        score = self.pool_score(torch.cat([h, stat], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        enc = (weights * h).sum(dim=2)
        last_stat = stat[:, :, -1, :]
        te_f = self.future_te(y_t).unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        tau = (y_t[:, :, 0] - t[:, -1, 0].view(batch, 1)).clamp_min(0.0)
        enc_exp = enc.unsqueeze(2).expand(-1, -1, horizon, -1)
        glob_exp = enc_exp.mean(dim=1, keepdim=True).expand(-1, self.n_vars, -1, -1)
        dec_support = torch.stack(
            [
                tau.unsqueeze(1).expand(-1, self.n_vars, -1),
                last_stat[..., 0].unsqueeze(-1).expand(-1, -1, horizon),
                last_stat[..., 2].unsqueeze(-1).expand(-1, -1, horizon),
                last_stat[..., 3].unsqueeze(-1).expand(-1, -1, horizon),
                last_stat[..., 5].unsqueeze(-1).expand(-1, -1, horizon),
                last_stat[..., 6].unsqueeze(-1).expand(-1, -1, horizon),
                last_stat[..., 9].unsqueeze(-1).expand(-1, -1, horizon),
                last_stat[..., 10].unsqueeze(-1).expand(-1, -1, horizon),
            ],
            dim=-1,
        )
        return torch.cat([enc_exp, glob_exp, te_f, dec_support], dim=-1), enc, last_stat

    @staticmethod
    def _masked_affine_anchor(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Normalize each sample-variable history and retain its affine anchor."""
        count = mask.sum(dim=1)
        denom = count.clamp_min(1.0)
        mean = (x * mask).sum(dim=1) / denom
        centered = (x - mean.unsqueeze(1)) * mask
        variance = centered.square().sum(dim=1) / denom
        scale = torch.sqrt(variance + 1e-4).clamp_min(0.05)
        observed = count > 0
        mean = torch.where(observed, mean, torch.zeros_like(mean))
        scale = torch.where(observed, scale, torch.ones_like(scale))
        normalized = centered / scale.unsqueeze(1)
        return normalized, mean, scale

    @staticmethod
    def _wearable_pose_groups(
        dataset_name: str,
    ) -> tuple[tuple[tuple[int, int, int], tuple[tuple[int, int, int], ...]], ...]:
        """Return (reference accelerometer, co-located vector groups)."""
        name = dataset_name.upper().replace("-", "")
        if name == "MHEALTH":
            return (
                ((0, 1, 2), ((0, 1, 2),)),
                ((5, 6, 7), ((5, 6, 7), (8, 9, 10), (11, 12, 13))),
                ((14, 15, 16), ((14, 15, 16), (17, 18, 19), (20, 21, 22))),
            )
        if name == "USCHAD":
            return (((0, 1, 2), ((0, 1, 2), (3, 4, 5))),)
        return ()

    @staticmethod
    def _pose_frame(reference: Tensor, reference_mask: Tensor, strict: bool) -> tuple[Tensor, Tensor]:
        """Estimate an equivariant orthogonal frame from a history accelerometer."""
        full = (reference_mask.min(dim=-1).values > 0.5).to(reference.dtype)
        count = full.sum(dim=1).clamp_min(1.0)
        mean = (reference * full.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1)
        centered = (reference - mean.unsqueeze(1)) * full.unsqueeze(-1)
        covariance = torch.einsum("bti,btj->bij", centered, centered) / count[:, None, None]

        eps = 1e-6
        mean_norm = torch.linalg.vector_norm(mean, dim=-1)
        e1 = mean / mean_norm.clamp_min(eps).unsqueeze(-1)
        cov_e1 = torch.matmul(covariance, e1.unsqueeze(-1)).squeeze(-1)
        transverse = cov_e1 - (cov_e1 * e1).sum(dim=-1, keepdim=True) * e1
        transverse_norm = torch.linalg.vector_norm(transverse, dim=-1)
        e2 = transverse / transverse_norm.clamp_min(eps).unsqueeze(-1)
        e3 = torch.linalg.cross(e1, e2, dim=-1)
        frame = torch.stack([e1, e2, e3], dim=-1)

        energy = ((reference.square().sum(dim=-1) * full).sum(dim=1) / count).sqrt()
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        gravity_ratio = mean_norm / energy.clamp_min(eps)
        transverse_ratio = transverse_norm / trace.clamp_min(eps)
        if strict:
            usable = (count >= 8) & (gravity_ratio >= 0.12) & (transverse_ratio >= 0.01)
        else:
            usable = (count >= 4) & (gravity_ratio >= 0.03) & (transverse_ratio >= 1e-4)
        identity = torch.eye(3, dtype=reference.dtype, device=reference.device).expand(reference.size(0), -1, -1)
        frame = torch.where(usable[:, None, None], frame, identity)
        return frame, usable.to(reference.dtype)

    def _canonicalize_pose(self, x: Tensor, mask: Tensor) -> Tensor:
        """Map co-located vector sensors into a history-derived wearable frame."""
        self._pose_frames = []
        usage: list[Tensor] = []
        if not self.pose_groups:
            self._pose_frame_usage = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x

        canonical = x.clone()
        for reference_idx, target_groups in self.pose_groups:
            reference = x[..., list(reference_idx)]
            reference_mask = mask[..., list(reference_idx)]
            frame, usable = self._pose_frame(reference, reference_mask, self.use_pose_reliable)
            usage.append(usable)
            for target_idx in target_groups:
                values = x[..., list(target_idx)]
                full = mask[..., list(target_idx)].min(dim=-1).values > 0.5
                rotated = torch.matmul(values, frame)
                canonical[..., list(target_idx)] = torch.where(full.unsqueeze(-1), rotated, values)
                self._pose_frames.append((target_idx, frame))
        self._pose_frame_usage = torch.stack(usage, dim=-1).mean(dim=-1)
        return canonical

    def _restore_pose(self, pred: Tensor) -> Tensor:
        """Map canonical vector forecasts back to the deployed sensor axes."""
        restored = pred.clone()
        for target_idx, frame in self._pose_frames:
            values = pred[..., list(target_idx)]
            # Keep the geometric inverse in FP32 under mixed-precision training;
            # otherwise autocast can produce a bfloat16 indexed-assignment source.
            with torch.autocast(device_type=pred.device.type, enabled=False):
                rotated = torch.matmul(values.float(), frame.float().transpose(-1, -2))
            restored[..., list(target_idx)] = rotated.to(dtype=restored.dtype)
        return restored

    def _ordered_patch_forecast(self, h: Tensor) -> Tensor:
        """Forecast from the complete ordered patch trajectory with a shared head."""
        if self.ordered_patch_head is None or self.ordered_patch_norm is None:
            raise RuntimeError("ordered patch modules are unavailable")
        if h.size(2) < self.ordered_n_patch:
            h = F.pad(h, (0, 0, 0, self.ordered_n_patch - h.size(2)))
        elif h.size(2) > self.ordered_n_patch:
            h = h[:, :, -self.ordered_n_patch :, :]
        trajectory = self.ordered_patch_norm(h).flatten(start_dim=2)
        return self.ordered_patch_head(trajectory).permute(0, 2, 1)

    def _lowrank_trajectory_forecast(self, h: Tensor) -> Tensor:
        """Forecast from an amplitude anchor and a shared low-rank patch path."""
        if (
            self.trajectory_head is None
            or self.trajectory_norm is None
            or self.trajectory_basis is None
        ):
            raise RuntimeError("low-rank trajectory modules are unavailable")
        if h.size(2) < self.trajectory_n_patch:
            h = F.pad(h, (0, 0, 0, self.trajectory_n_patch - h.size(2)))
        elif h.size(2) > self.trajectory_n_patch:
            h = h[:, :, -self.trajectory_n_patch :, :]
        h = self.trajectory_norm(h)
        anchor = h[:, :, -1, :]
        displacement = h - anchor.unsqueeze(2)
        basis = self.trajectory_basis.to(device=h.device, dtype=h.dtype)
        coefficient = torch.einsum("bvpd,pr->bvrd", displacement, basis)
        coordinate = torch.cat([anchor, coefficient.flatten(start_dim=2)], dim=-1)
        return self.trajectory_head(coordinate).permute(0, 2, 1)

    def _spectral_trajectory_context(
        self,
        h: Tensor,
        stat: Tensor,
        future_te: Tensor,
        tau: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Retrieve horizon-specific context from population-shrunk DCT tokens."""
        modules = (
            self.spectral_norm,
            self.spectral_frequency,
            self.spectral_query,
            self.spectral_key,
            self.spectral_value,
            self.spectral_context_norm,
        )
        if self.spectral_basis is None or any(module is None for module in modules):
            raise RuntimeError("spectral trajectory modules are unavailable")

        if h.size(2) < self.spectral_n_patch:
            missing = self.spectral_n_patch - h.size(2)
            h = F.pad(h, (0, 0, missing, 0))
            stat = F.pad(stat, (0, 0, missing, 0))
        elif h.size(2) > self.spectral_n_patch:
            h = h[:, :, -self.spectral_n_patch :, :]
            stat = stat[:, :, -self.spectral_n_patch :, :]

        h = self.spectral_norm(h)
        anchor = h[:, :, -1, :]
        displacement = h - anchor.unsqueeze(2)
        basis = self.spectral_basis.to(device=h.device, dtype=h.dtype)
        coefficient = torch.einsum("bvpd,pk->bvkd", displacement, basis)

        basis_energy = basis.square()
        support_state = torch.einsum("bvpc,pk->bvkc", stat, basis_energy)
        coefficient_energy = torch.log1p(coefficient.square().mean(dim=-1, keepdim=True))
        if self.use_spectral_support_only:
            coefficient_energy = torch.zeros_like(coefficient_energy)
        frequency = torch.linspace(
            0.0,
            1.0,
            self.spectral_n_patch,
            device=h.device,
            dtype=h.dtype,
        ).view(1, 1, self.spectral_n_patch, 1)
        frequency = frequency.expand(h.size(0), h.size(1), -1, -1)

        if self.use_spectral_global:
            if self.spectral_global_logit is None:
                raise RuntimeError("global spectral gate is unavailable")
            gate = torch.sigmoid(self.spectral_global_logit).expand(
                h.size(0), h.size(1), -1, -1
            )
        else:
            if self.spectral_gate is None:
                raise RuntimeError("sample-conditioned spectral gate is unavailable")
            gate_input = torch.cat([support_state, coefficient_energy, frequency], dim=-1)
            gate = torch.sigmoid(self.spectral_gate(gate_input))

        gated_coefficient = gate * coefficient
        key = self.spectral_key(gated_coefficient) + self.spectral_frequency(frequency)
        value = self.spectral_value(gated_coefficient)
        horizon = future_te.size(1)
        query_input = torch.cat(
            [
                anchor.unsqueeze(2).expand(-1, -1, horizon, -1),
                future_te.unsqueeze(1).expand(-1, h.size(1), -1, -1),
                tau[:, None, :, None].expand(-1, h.size(1), -1, -1),
            ],
            dim=-1,
        )
        query = self.spectral_query(query_input)
        logits = torch.einsum("bvhd,bvkd->bvhk", query, key) / math.sqrt(self.d_model)
        weight = torch.softmax(logits, dim=-1)
        retrieved = torch.einsum("bvhk,bvkd->bvhd", weight, value)
        context = self.spectral_context_norm(anchor.unsqueeze(2) + retrieved)
        return context, gate, weight

    def _waveform_coordinate(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the normalized within-patch waveform before statistic compression."""
        if self.waveform_proj is None or self.waveform_norm is None:
            raise RuntimeError("waveform coordinate requested without initialized modules")
        x = self._pad(x, 0.0)
        mask = self._pad(mask, 0.0)
        batch, padded_len, n_vars = x.shape
        n_patch = padded_len // self.patch_len
        x_patch = x.view(batch, n_patch, self.patch_len, n_vars).permute(0, 3, 1, 2)
        m_patch = mask.view(batch, n_patch, self.patch_len, n_vars).permute(0, 3, 1, 2)
        values = x_patch * m_patch
        pair_mask = m_patch[..., 1:] * m_patch[..., :-1]
        velocity = (x_patch[..., 1:] - x_patch[..., :-1]) * pair_mask
        velocity = F.pad(velocity, (1, 0))
        var = self.var_emb.to(x.device, x.dtype).view(1, n_vars, 1, self.d_model)
        var = var.expand(batch, -1, n_patch, -1)
        coordinate = self.waveform_proj(torch.cat([values, m_patch, velocity, var], dim=-1))
        return self.waveform_norm(coordinate), (m_patch.sum(dim=-1) > 0).to(x.dtype)

    def _waveform_cross_attention(
        self,
        anchor: Tensor,
        stat: Tensor,
        waveform: Tensor,
        waveform_obs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Inject ordered waveform evidence without changing the patchwise level."""
        if (
            self.waveform_cross_attn is None
            or self.waveform_cross_gate is None
            or self.waveform_cross_norm is None
        ):
            raise RuntimeError("waveform cross-attention modules are unavailable")
        batch, n_vars, n_patch, width = anchor.shape
        query = anchor.reshape(batch * n_vars, n_patch, width)
        source = waveform.reshape(batch * n_vars, n_patch, width)
        valid = waveform_obs.reshape(batch * n_vars, n_patch) > 0
        key_padding_mask = ~valid
        all_missing = ~valid.any(dim=1)
        if all_missing.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_missing, 0] = False
        update, _ = self.waveform_cross_attn(
            query,
            source,
            source,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        update = update.reshape(batch, n_vars, n_patch, width)
        obs = waveform_obs.unsqueeze(-1).to(dtype=update.dtype)
        center = (update * obs).sum(dim=2, keepdim=True) / obs.sum(dim=2, keepdim=True).clamp_min(1.0)
        update = (update - center) * obs
        gate = torch.sigmoid(self.waveform_cross_gate(torch.cat([stat, anchor, waveform], dim=-1)))
        return self.waveform_cross_norm(anchor + gate * update), gate

    def _level_shape_forecast(
        self,
        fixed_token: Tensor,
        fixed_stat: Tensor,
        waveform_token: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forecast with disjoint level and zero-mean waveform coordinates."""
        if (
            self.level_var_attn is None
            or self.shape_var_attn is None
            or self.level_var_norm is None
            or self.shape_var_norm is None
            or self.level_shape_level_head is None
            or self.level_shape_shape_head is None
        ):
            raise RuntimeError("level/shape decomposition modules are unavailable")

        level_token = fixed_token[:, :, -1, :]
        level_context, _ = self.level_var_attn(level_token, level_token, level_token, need_weights=False)
        level_context = self.level_var_norm(level_token + level_context)
        level = self.level_shape_level_head(
            torch.cat([level_context, fixed_stat[:, :, -1, :]], dim=-1)
        ).squeeze(-1)
        level = level.unsqueeze(1).expand(-1, self.pred_len, -1)

        if waveform_token.size(2) < self.level_shape_n_patch:
            waveform_token = F.pad(
                waveform_token,
                (0, 0, 0, self.level_shape_n_patch - waveform_token.size(2)),
            )
        elif waveform_token.size(2) > self.level_shape_n_patch:
            waveform_token = waveform_token[:, :, -self.level_shape_n_patch :, :]
        batch, n_vars, n_patch, width = waveform_token.shape
        shape_token = waveform_token.permute(0, 2, 1, 3).reshape(batch * n_patch, n_vars, width)
        shape_context, _ = self.shape_var_attn(shape_token, shape_token, shape_token, need_weights=False)
        shape_token = self.shape_var_norm(shape_token + shape_context)
        shape_token = shape_token.reshape(batch, n_patch, n_vars, width).permute(0, 2, 1, 3)
        shape = self.level_shape_shape_head(shape_token.flatten(start_dim=2)).permute(0, 2, 1)
        shape = shape - shape.mean(dim=1, keepdim=True)
        return level + shape, level, shape

    def _dct_dual_forecast(
        self,
        level_token: Tensor,
        waveform_token: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forecast orthogonal low- and high-frequency horizon coordinates."""
        modules = (
            self.dual_dct_basis,
            self.dual_level_conv,
            self.dual_shape_conv,
            self.dual_level_mix,
            self.dual_shape_mix,
            self.dual_level_var,
            self.dual_shape_var,
            self.dual_level_norm,
            self.dual_shape_norm,
            self.dual_level_head,
            self.dual_shape_head,
        )
        if any(module is None for module in modules):
            raise RuntimeError("DCT dual-coordinate modules are unavailable")

        def align(token: Tensor) -> Tensor:
            if token.size(2) < self.dual_n_patch:
                return F.pad(token, (0, 0, 0, self.dual_n_patch - token.size(2)))
            return token[:, :, -self.dual_n_patch :, :]

        level_token = align(level_token)
        waveform_token = align(waveform_token)
        batch, n_vars, n_patch, width = level_token.shape

        level_seq = level_token.reshape(batch * n_vars, n_patch, width).transpose(1, 2)
        shape_seq = waveform_token.reshape(batch * n_vars, n_patch, width).transpose(1, 2)
        if not self.dual_no_causal:
            level_update = self.dual_level_mix(F.gelu(self.dual_level_conv(level_seq)[..., :n_patch]))
            shape_update = self.dual_shape_mix(F.gelu(self.dual_shape_conv(shape_seq)[..., :n_patch]))
            level_seq = level_seq + level_update
            shape_seq = shape_seq + shape_update
        level_seq = level_seq.transpose(1, 2).reshape(batch, n_vars, n_patch, width)
        shape_seq = shape_seq.transpose(1, 2).reshape(batch, n_vars, n_patch, width)

        level_var = level_seq.permute(0, 2, 1, 3).reshape(batch * n_patch, n_vars, width)
        shape_var = shape_seq.permute(0, 2, 1, 3).reshape(batch * n_patch, n_vars, width)
        level_context, _ = self.dual_level_var(level_var, level_var, level_var, need_weights=False)
        shape_context, _ = self.dual_shape_var(shape_var, shape_var, shape_var, need_weights=False)
        if self.dual_level_capacity is not None and self.dual_shape_capacity is not None:
            level_context = level_context + self.dual_level_capacity(level_var)
            shape_context = shape_context + self.dual_shape_capacity(shape_var)
        level_var = self.dual_level_norm(level_var + level_context)
        shape_var = self.dual_shape_norm(shape_var + shape_context)
        level_seq = level_var.reshape(batch, n_patch, n_vars, width).permute(0, 2, 1, 3)
        shape_seq = shape_var.reshape(batch, n_patch, n_vars, width).permute(0, 2, 1, 3)

        low_coefficient = self.dual_level_head(level_seq.flatten(start_dim=2))
        low = torch.einsum("bvr,tr->bvt", low_coefficient, self.dual_dct_basis)
        high_raw = self.dual_shape_head(shape_seq.flatten(start_dim=2))
        if self.dual_no_high:
            high = torch.zeros_like(high_raw)
        elif self.dual_low_subspace_only:
            # Parameter-matched single-coordinate control: both prediction
            # branches remain active, but their sum is confined to the same
            # low-frequency DCT subspace.
            leaked_low = torch.einsum("bvt,tr->bvr", high_raw, self.dual_dct_basis)
            high = torch.einsum("bvr,tr->bvt", leaked_low, self.dual_dct_basis)
        elif self.dual_no_orthogonal:
            high = high_raw
        else:
            leaked_low = torch.einsum("bvt,tr->bvr", high_raw, self.dual_dct_basis)
            low_projection = torch.einsum("bvr,tr->bvt", leaked_low, self.dual_dct_basis)
            if self.dual_fixed_orth_weight is not None:
                high = high_raw - self.dual_fixed_orth_weight * low_projection
            elif self.dual_soft_orthogonal:
                if self.dual_orth_logit is None:
                    raise RuntimeError("Soft orthogonality parameter is unavailable")
                high = high_raw - torch.sigmoid(self.dual_orth_logit) * low_projection
            else:
                high = high_raw - low_projection
        pred = (low + high).permute(0, 2, 1)
        return (
            pred,
            low.permute(0, 2, 1),
            high.permute(0, 2, 1),
            high_raw.permute(0, 2, 1),
        )

    def _horizon_patch_context(
        self,
        h: Tensor,
        stat: Tensor,
        observed: Tensor,
        pooled: Tensor,
        future_te: Tensor,
        tau: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Retrieve horizon-specific dynamics from the ordered patch memory."""
        if self.hpq_query is None or self.hpq_key is None or self.hpq_value is None:
            raise RuntimeError("horizon patch-query modules are unavailable")
        batch, n_vars, n_patch, _ = h.shape
        horizon = future_te.size(1)
        last = h[:, :, -1, :]
        query_in = torch.cat(
            [
                last.unsqueeze(2).expand(-1, -1, horizon, -1),
                future_te.unsqueeze(1).expand(-1, n_vars, -1, -1),
                tau[:, None, :, None].expand(-1, n_vars, -1, -1),
            ],
            dim=-1,
        )
        query = self.hpq_query(query_in)
        key = self.hpq_key(h)
        value = self.hpq_value(h)
        logits = torch.einsum("bvhd,bvpd->bvhp", query, key) / math.sqrt(self.d_model)
        logits = logits.masked_fill(observed[:, :, None, :] <= 0, -1e4)
        weight = torch.softmax(logits, dim=-1)
        retrieved = torch.einsum("bvhp,bvpd->bvhd", weight, value)
        base = pooled.unsqueeze(2).expand(-1, -1, horizon, -1)
        last_stat = stat[:, :, -1, :].unsqueeze(2).expand(-1, -1, horizon, -1)
        gate = torch.sigmoid(self.hpq_gate(torch.cat([last_stat, base, query, retrieved], dim=-1)))
        context = self.hpq_norm(base + gate * (retrieved - base))
        return context, gate

    @staticmethod
    def _standard_revin_anchor(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Standard time-axis RevIN on the zero-filled irregular tensor."""
        mean = x.mean(dim=1)
        centered = x - mean.unsqueeze(1)
        scale = torch.sqrt(centered.square().mean(dim=1) + 1e-4).clamp_min(0.05)
        return centered / scale.unsqueeze(1), mean, scale

    def _masked_revin_anchor(
        self, x: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Mask-aware RevIN with learnable affine parameters."""
        normalized, mean, scale = self._masked_affine_anchor(x, mask)
        normalized = normalized * self.masked_revin_weight + self.masked_revin_bias
        return normalized, mean, scale

    def _restore_anchor(self, pred: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        if self.use_masked_revin:
            weight = self.masked_revin_weight.clamp_min(1e-4)
            pred = (pred - self.masked_revin_bias) / weight
        return pred * scale.unsqueeze(1) + mean.unsqueeze(1)

    @staticmethod
    def _masked_patch_pool(score: Tensor, observed: Tensor) -> Tensor:
        score = score.masked_fill(observed <= 0, -1e4)
        return torch.softmax(score, dim=-1)

    @staticmethod
    def _history_context(x: Tensor, mask: Tensor, t: Tensor) -> Tensor:
        """Deployment-visible sample context for unseen-subject adaptation."""
        mask = mask.to(dtype=x.dtype)
        count = mask.sum(dim=(1, 2)).clamp_min(1.0)
        density = mask.mean(dim=(1, 2))
        coverage = (mask.sum(dim=1) > 0).to(dtype=x.dtype).mean(dim=1)
        abs_mean = (x.abs() * mask).sum(dim=(1, 2)) / count
        rms = torch.sqrt((x.square() * mask).sum(dim=(1, 2)) / count + 1e-6)

        if x.size(1) > 1:
            pair_mask = mask[:, 1:, :] * mask[:, :-1, :]
            pair_count = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
            delta_abs = ((x[:, 1:, :] - x[:, :-1, :]).abs() * pair_mask).sum(dim=(1, 2)) / pair_count
            dt = (t[:, 1:, 0] - t[:, :-1, 0]).clamp_min(0.0)
            dt_mean = dt.mean(dim=1)
            dt_cv = dt.std(dim=1, unbiased=False) / dt_mean.clamp_min(1e-6)
        else:
            delta_abs = torch.zeros_like(density)
            dt_mean = torch.zeros_like(density)
            dt_cv = torch.zeros_like(density)

        indices = torch.arange(x.size(1), device=x.device).view(1, -1, 1)
        last_index = torch.where(mask > 0, indices, torch.full_like(indices, -1)).amax(dim=1)
        safe_index = last_index.clamp_min(0).long().unsqueeze(1)
        last_value = x.gather(1, safe_index).squeeze(1)
        has_variable = (last_index >= 0).to(dtype=x.dtype)
        last_abs = (last_value.abs() * has_variable).sum(dim=1) / has_variable.sum(dim=1).clamp_min(1.0)

        return torch.stack(
            [density, coverage, abs_mean, rms, delta_abs, last_abs, dt_mean, dt_cv],
            dim=-1,
        ).clamp(-10.0, 10.0)

    def _subject_robust_aux(
        self,
        pred: Tensor,
        true: Tensor,
        mask: Tensor,
        subject_id: Tensor | None,
        subject_weight: Tensor | None,
    ) -> Tensor | None:
        """Replace window-average MSE with subject-balanced or soft worst-group risk."""
        if not self.training or not self.use_subject_balancing or subject_id is None:
            return None
        mask = mask.to(dtype=pred.dtype)
        sample_num = ((pred - true).square() * mask).flatten(1).sum(dim=1)
        sample_den = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        base = sample_num.sum() / sample_den.sum().clamp_min(1.0)
        if subject_weight is None:
            subject_weight = torch.ones_like(sample_num)
        subject_weight = subject_weight.to(device=pred.device, dtype=pred.dtype).flatten()
        balanced = (subject_weight * sample_num).sum() / (subject_weight * sample_den).sum().clamp_min(1.0)
        target = balanced
        if self.use_subject_groupdro:
            subject_id = subject_id.to(device=pred.device).flatten()
            group_risks = []
            for group in torch.unique(subject_id):
                selected = subject_id == group
                group_risks.append(sample_num[selected].sum() / sample_den[selected].sum().clamp_min(1.0))
            if len(group_risks) > 1:
                risks = torch.stack(group_risks)
                weights = torch.softmax(risks.detach() / self.subject_dro_temperature, dim=0)
                robust = (weights * risks).sum()
                target = (1.0 - self.subject_dro_mix) * balanced + self.subject_dro_mix * robust
        return target - base

    def _horizon_simplex_forecast(
        self,
        fixed_token: Tensor,
        fixed_stat: Tensor,
        fixed_obs: Tensor,
        fixed_alias: Tensor,
        y_t: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode from a fixed-anchor plus multiresolution scale simplex.

        Scale candidates remain separate until a future-time query is known.
        The fixed coordinate is candidate zero, so the simplex can recover the
        local anchor without a second admission gate.
        """
        if self._cag_scale_tokens is None or self._ct_scale_stat is None:
            raise RuntimeError("Horizon simplex requires the multiresolution scale bank.")

        batch, n_vars, n_patch, dim = fixed_token.shape
        n_scales = self._cag_scale_tokens.size(2)
        fixed_h = AMPGModel._temporal_graph(self, fixed_token, fixed_stat, fixed_alias)
        fixed_score = self.pool_score(torch.cat([fixed_h, fixed_stat], dim=-1)).squeeze(-1)
        fixed_weight = self._masked_patch_pool(fixed_score, fixed_obs).unsqueeze(-1)
        fixed_enc = (fixed_weight * fixed_h).sum(dim=2)
        fixed_summary = (fixed_weight * fixed_stat).sum(dim=2)

        scale_token = self._cag_scale_tokens[:, :, :, :n_patch, :]
        scale_stat = self._prepare_support_stat(self._ct_scale_stat[:, :, :, :n_patch, :])
        scale_obs = scale_stat[..., 1]

        # The shared temporal operator is applied independently to every scale;
        # scale identity is preserved until the future-query simplex below.
        scale_seq = scale_token.permute(0, 2, 1, 3, 4).reshape(batch * n_scales * n_vars, n_patch, dim)
        pos = self.temporal_pos[:, :n_patch, :].to(scale_seq.device, scale_seq.dtype)
        scale_seq = self.temporal(scale_seq + pos)
        scale_h = scale_seq.view(batch, n_scales, n_vars, n_patch, dim).permute(0, 2, 1, 3, 4)
        scale_score = self.horizon_scale_pool(torch.cat([scale_h, scale_stat], dim=-1)).squeeze(-1)
        scale_weight = self._masked_patch_pool(scale_score, scale_obs).unsqueeze(-1)
        scale_enc = (scale_weight * scale_h).sum(dim=3)
        scale_summary = (scale_weight * scale_stat).sum(dim=3)

        candidates = torch.cat([fixed_enc.unsqueeze(2), scale_enc], dim=2)
        candidate_stat = torch.cat([fixed_summary.unsqueeze(2), scale_summary], dim=2)

        future_te = self.future_te(y_t)
        tau = (y_t[:, :, 0] - t[:, -1, 0].view(batch, 1)).clamp_min(0.0)
        query = self.horizon_query(torch.cat([future_te, tau.unsqueeze(-1)], dim=-1))
        logits = torch.einsum(
            "bvsd,bhd->bvhs",
            self.horizon_scale_key(candidates),
            query,
        ) / math.sqrt(max(dim, 1))

        last_fixed = fixed_stat[:, :, -1, :]
        state = last_fixed.unsqueeze(2).expand(-1, -1, y_t.size(1), -1)
        future_state = future_te.unsqueeze(1).expand(-1, n_vars, -1, -1)
        tau_state = tau.unsqueeze(1).unsqueeze(-1).expand(-1, n_vars, -1, -1)
        logits = logits + self.horizon_state_bias(torch.cat([state, future_state, tau_state], dim=-1))

        density = candidate_stat[..., 0].clamp(0.05, 1.0)
        reliability_price = F.softplus(self.horizon_reliability_logit)
        if "no_reliability_price" in self.mode:
            reliability_price = torch.zeros_like(reliability_price)
        logits = logits - reliability_price * (density.rsqrt() - 1.0).unsqueeze(2)
        simplex = torch.softmax(logits, dim=-1)

        enc_h = torch.einsum("bvhs,bvsd->bvhd", simplex, candidates)
        stat_h = torch.einsum("bvhs,bvsc->bvhc", simplex, candidate_stat)
        enc_h = self.horizon_simplex_norm(enc_h)

        if "no_graph" not in self.mode:
            q = self.graph_q(enc_h).permute(0, 2, 1, 3)
            k = self.graph_k(enc_h).permute(0, 2, 1, 3)
            attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1)), dim=-1)
            msg = torch.einsum("bhij,bjhd->bihd", attn, enc_h)
            msg = self.graph_msg(msg)
            graph_stat = torch.zeros_like(stat_h) if "no_q_graph" in self.mode else stat_h
            graph_gate = torch.sigmoid(self.graph_gate(torch.cat([enc_h, msg, graph_stat], dim=-1)))
            enc_h = self.graph_norm(enc_h + graph_gate * msg)

        global_h = enc_h.mean(dim=1, keepdim=True).expand(-1, n_vars, -1, -1)
        future_h = future_te.unsqueeze(1).expand(-1, n_vars, -1, -1)
        dec_support = torch.stack(
            [
                tau.unsqueeze(1).expand(-1, n_vars, -1),
                stat_h[..., 0],
                stat_h[..., 2],
                stat_h[..., 3],
                stat_h[..., 5],
                stat_h[..., 6],
                stat_h[..., 9],
                stat_h[..., 10],
            ],
            dim=-1,
        )
        dec_in = torch.cat([enc_h, global_h, future_h, dec_support], dim=-1)
        pred = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)
        admitted = 1.0 - simplex[..., :1]
        return pred, stat_h, admitted, simplex

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
        if self.use_pose_canonical:
            x = self._canonicalize_pose(x, x_mask)
        anchor_mean = None
        anchor_scale = None
        if self.use_masked_revin:
            x, anchor_mean, anchor_scale = self._masked_revin_anchor(x, x_mask)
        elif self.use_standard_revin:
            x, anchor_mean, anchor_scale = self._standard_revin_anchor(x)
        elif self.use_affine_anchor:
            x, anchor_mean, anchor_scale = self._masked_affine_anchor(x, x_mask)
        t = self._get_time(x_mark, x)
        history_context = self._history_context(x, x_mask, t) if self.use_subject_context else None
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
        has_obs = torch.maximum(fixed_obs, scale_obs)
        waveform_gate = None
        waveform_cross_gate = None
        waveform_ordered_source = None
        waveform_token = None
        if self.use_waveform_token:
            if self.waveform_gate is None:
                raise RuntimeError("waveform gate is unavailable")
            waveform_token, waveform_obs = self._waveform_coordinate(x, x_mask)
            waveform_token = waveform_token[:, :, :n_patch, :]
            waveform_obs = waveform_obs[:, :, :n_patch]
            waveform_gate = torch.sigmoid(
                self.waveform_gate(torch.cat([fixed_stat, fixed_token, waveform_token], dim=-1))
            )
            waveform_gate = waveform_gate * waveform_obs.unsqueeze(-1)
            if self.use_waveform_crossattn:
                fixed_token, waveform_cross_gate = self._waveform_cross_attention(
                    fixed_token,
                    fixed_stat,
                    waveform_token,
                    waveform_obs,
                )
            elif self.use_parallel_waveform:
                # Keep the value-anchor path unchanged. The ordered decoder
                # receives waveform coordinates through a separate predictive
                # path and combines them only at the bounded output fusion.
                waveform_ordered_source = waveform_token
            elif self.use_waveform_residual:
                # Preserve the fixed value anchor and admit waveform evidence as
                # a closed residual coordinate inside the backbone token.
                fixed_token = fixed_token + waveform_gate * waveform_token
            else:
                fixed_token = fixed_token + waveform_gate * (waveform_token - fixed_token)

        if self.use_level_shape_decomp:
            if waveform_token is None:
                raise RuntimeError("level/shape decomposition requires waveform coordinates")
            pred, level_pred, shape_pred = self._level_shape_forecast(
                fixed_token,
                fixed_stat,
                waveform_token,
            )
            if anchor_mean is not None and anchor_scale is not None and not self.disable_anchor_restore:
                pred = self._restore_anchor(pred, anchor_mean, anchor_scale)
                level_pred = self._restore_anchor(level_pred, anchor_mean, anchor_scale)
                shape_pred = pred - level_pred
            if self.use_pose_canonical:
                pred = self._restore_pose(pred)
                level_pred = self._restore_pose(level_pred)
                shape_pred = pred - level_pred
            f_dim = -1 if self.configs.features == "MS" else 0
            pred = pred[:, -y.shape[1]:, f_dim:]
            level_pred = level_pred[:, -y.shape[1]:, f_dim:]
            shape_pred = shape_pred[:, -y.shape[1]:, f_dim:]
            true = y[:, :, f_dim:]
            mask = y_mask[:, :, f_dim:].to(dtype=pred.dtype)
            out = {"pred": pred, "true": true, "mask": mask}
            if self.training:
                count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                target_level = (true * mask).sum(dim=1, keepdim=True) / count
                target_level = target_level.expand_as(true)
                target_shape = true - target_level
                level_error = level_pred - target_level
                shape_error = shape_pred - target_shape
                level_loss = ((level_error.square() + 0.5 * level_error.abs()) * mask).sum()
                shape_loss = ((shape_error.square() + 0.5 * shape_error.abs()) * mask).sum()
                denom = mask.sum().clamp_min(1.0)
                out["training_loss"] = 0.5 * (level_loss + shape_loss) / denom
            out["level_shape_level"] = level_pred.detach()
            out["level_shape_shape"] = shape_pred.detach()
            return out

        if self.use_dct_dual_coordinate:
            if waveform_token is None:
                raise RuntimeError("DCT dual-coordinate forecasting requires waveform tokens")
            if self.dual_fixed_level:
                level_token = fixed_token
                stat = fixed_stat
                admit = torch.zeros_like(fixed_obs).unsqueeze(-1)
            else:
                level_token, stat, admit = self._admit(
                    fixed_token,
                    fixed_stat,
                    scale_token,
                    scale_stat,
                )
            pred, low_pred, high_pred, high_raw_pred = self._dct_dual_forecast(
                level_token, waveform_token
            )
            if anchor_mean is not None and anchor_scale is not None and not self.disable_anchor_restore:
                pred = self._restore_anchor(pred, anchor_mean, anchor_scale)
                low_pred = self._restore_anchor(low_pred, anchor_mean, anchor_scale)
                high_pred = pred - low_pred
                high_raw_pred = high_raw_pred * anchor_scale.unsqueeze(1)
            if self.use_pose_canonical:
                pred = self._restore_pose(pred)
                low_pred = self._restore_pose(low_pred)
                high_pred = pred - low_pred
                high_raw_pred = self._restore_pose(high_raw_pred)
            f_dim = -1 if self.configs.features == "MS" else 0
            return {
                "pred": pred[:, -y.shape[1]:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
                "dual_low": low_pred[:, -y.shape[1]:, f_dim:].detach(),
                "dual_high": high_pred[:, -y.shape[1]:, f_dim:].detach(),
                "dual_high_raw": high_raw_pred[:, -y.shape[1]:, f_dim:].detach(),
                "dual_orth_weight": (
                    torch.sigmoid(self.dual_orth_logit).detach()
                    if self.dual_orth_logit is not None
                    else pred.new_tensor(
                        self.dual_fixed_orth_weight
                        if self.dual_fixed_orth_weight is not None
                        else (1.0 if not self.dual_no_orthogonal else 0.0)
                    )
                ),
                "mesa_admit": admit.detach(),
                "mesa_density": stat[..., 0].detach(),
                "mesa_recency": stat[..., 10].detach(),
            }

        if self.use_horizon_simplex:
            pred, stat_h, admit, horizon_simplex = self._horizon_simplex_forecast(
                fixed_token,
                fixed_stat,
                fixed_obs,
                fixed_alias,
                y_t,
                t,
            )
            if anchor_mean is not None and anchor_scale is not None and not self.disable_anchor_restore:
                pred = self._restore_anchor(pred, anchor_mean, anchor_scale)
            if self.use_pose_canonical:
                pred = self._restore_pose(pred)
            f_dim = -1 if self.configs.features == "MS" else 0
            out = {
                "pred": pred[:, -y.shape[1]:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
                "mesa_admit": admit.detach(),
                "mesa_density": stat_h[..., 0].detach(),
                "mesa_obs": stat_h[..., 1].detach(),
                "mesa_recency": stat_h[..., 10].detach(),
                "mesa_span": stat_h[..., 9].detach(),
                "mesa_value_var": stat_h[..., 3].detach(),
                "mesa_coobs": stat_h[..., 1].detach(),
                "mesa_scale_alpha": horizon_simplex.detach(),
                "mesa_scale_width": self.ct_log_width.exp()
                .to(device=pred.device, dtype=pred.dtype)
                .unsqueeze(0)
                .expand(batch, -1)
                .detach(),
            }
            self.latest_diag = {
                "admit_mean": float(admit.mean().detach().cpu()),
                "fixed_anchor_mass": float(horizon_simplex[..., 0].mean().detach().cpu()),
                "reliability_price": float(F.softplus(self.horizon_reliability_logit).detach().cpu()),
            }
            return out

        decoder_position_admit = False
        if "ms_concat_param" in self.mode or "multiscale_concat" in self.mode:
            token, stat, admit = self._multiscale_concat(fixed_token, fixed_stat, scale_stat, n_patch)
            h = AMPGModel._temporal_graph(self, token, stat, fixed_alias)
        elif "decoder_level_admit" in self.mode:
            fixed_h = AMPGModel._temporal_graph(self, fixed_token, fixed_stat, fixed_alias)
            scale_h = AMPGModel._temporal_graph(self, scale_token, scale_stat, fixed_alias)
            _, stat, admit = self._admit(fixed_token, fixed_stat, scale_token, scale_stat)
            decoder_position_admit = True
            h = fixed_h
        elif "post_graph_admit" in self.mode:
            fixed_h = AMPGModel._temporal_graph(self, fixed_token, fixed_stat, fixed_alias)
            scale_h = AMPGModel._temporal_graph(self, scale_token, scale_stat, fixed_alias)
            h, stat, admit = self._admit(fixed_h, fixed_stat, scale_h, scale_stat)
        elif "post_temporal_admit" in self.mode:
            fixed_h = self._temporal_stage(fixed_token)
            scale_h = self._temporal_stage(scale_token)
            h, stat, admit = self._admit(fixed_h, fixed_stat, scale_h, scale_stat)
            h = self._variable_stage(h, stat)
        else:
            token, stat, admit = self._admit(fixed_token, fixed_stat, scale_token, scale_stat)
            h = AMPGModel._temporal_graph(self, token, stat, fixed_alias)
        if decoder_position_admit:
            fixed_h = self._decoder_position_tail(fixed_h, fixed_stat, history_context)
            scale_h = self._decoder_position_tail(scale_h, scale_stat, history_context)
            fixed_dec_in, fixed_enc, fixed_last = self._basic_decoder_input(
                fixed_h, fixed_stat, fixed_obs, y_t, t, y.size(1)
            )
            scale_dec_in, scale_enc, scale_last = self._basic_decoder_input(
                scale_h, scale_stat, scale_obs, y_t, t, y.size(1)
            )
            valid = has_obs.to(dtype=admit.dtype)
            decoder_admit = (admit.squeeze(-1) * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
            decoder_admit = decoder_admit.unsqueeze(-1).unsqueeze(-1)
            dec_in = fixed_dec_in + decoder_admit * (scale_dec_in - fixed_dec_in)
            enc_gate = decoder_admit.squeeze(-1)
            enc = fixed_enc + enc_gate * (scale_enc - fixed_enc)
            stat_gate = decoder_admit.squeeze(-1)
            last_stat = fixed_last + stat_gate * (scale_last - fixed_last)
            h = fixed_h + decoder_admit * (scale_h - fixed_h)
            te_f = self.future_te(y_t).unsqueeze(1).expand(-1, self.n_vars, -1, -1)
            tau = (y_t[:, :, 0] - t[:, -1, 0].view(batch, 1)).clamp_min(0.0)
        else:
            if "no_stconv" not in self.mode:
                msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                gate = torch.sigmoid(self.ct_st_gate(torch.cat([stat, h], dim=-1)))
                h = self.ct_st_norm(h + gate * msg)
            if history_context is not None:
                context_params = self.subject_context(history_context)
                gamma, beta = context_params.chunk(2, dim=-1)
                h = self.subject_context_norm(
                    (1.0 + self.subject_context_scale * torch.tanh(gamma)[:, None, None, :]) * h
                    + self.subject_context_scale * beta[:, None, None, :]
                )

            score = self.pool_score(torch.cat([h, stat], dim=-1)).squeeze(-1)
            score = score.masked_fill(has_obs <= 0, -1e4)
            weights = torch.softmax(score, dim=-1).unsqueeze(-1)
            enc = (weights * h).sum(dim=2)

            last_stat = stat[:, :, -1, :]
            te_f = self.future_te(y_t).unsqueeze(1).expand(-1, self.n_vars, -1, -1)
            tau = (y_t[:, :, 0] - t[:, -1, 0].view(batch, 1)).clamp_min(0.0)
        hpq_gate = None
        spectral_gate = None
        spectral_weight = None
        if decoder_position_admit:
            enc_exp = enc.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        elif self.use_spectral_trajectory:
            enc_exp, spectral_gate, spectral_weight = self._spectral_trajectory_context(
                h, stat, te_f[:, 0, :, :], tau
            )
        elif self.use_horizon_patch_query:
            enc_exp, hpq_gate = self._horizon_patch_context(
                h, stat, has_obs, enc, te_f[:, 0, :, :], tau
            )
        else:
            enc_exp = enc.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        glob_exp = enc_exp.mean(dim=1, keepdim=True).expand(-1, self.n_vars, -1, -1)
        if not decoder_position_admit:
            dec_support = torch.stack(
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
            dec_in = torch.cat([enc_exp, glob_exp, te_f, dec_support], dim=-1)
        pred = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)
        base_pred_for_route = pred
        ordered_pred_for_route = None
        ordered_logit_for_route = None
        ordered_gate = None
        if self.use_ordered_patch:
            ordered_source = waveform_ordered_source if waveform_ordered_source is not None else h
            ordered_pred = self._ordered_patch_forecast(ordered_source)
            ordered_pred_for_route = ordered_pred
            if self.use_ordered_patch_only:
                pred = ordered_pred
            else:
                if self.use_separable_ordered_patch:
                    if (
                        self.ordered_global_logit is None
                        or self.ordered_horizon_logit is None
                        or self.ordered_variable_logit is None
                    ):
                        raise RuntimeError("separable ordered gate is unavailable")
                    ordered_logit_for_route = (
                        self.ordered_global_logit
                        + self.ordered_horizon_logit
                        + self.ordered_variable_logit
                    ).expand(batch, -1, -1)
                    ordered_gate = torch.sigmoid(ordered_logit_for_route)
                else:
                    if self.ordered_patch_gate is None:
                        raise RuntimeError("ordered patch gate is unavailable")
                    ordered_logit = self.ordered_patch_gate(torch.cat([last_stat, enc], dim=-1))
                    if self.use_hierarchical_ordered_patch:
                        if self.ordered_global_logit is None:
                            raise RuntimeError("hierarchical ordered gate is unavailable")
                        ordered_logit = ordered_logit + self.ordered_global_logit
                    ordered_logit_for_route = ordered_logit.permute(0, 2, 1)
                    ordered_gate = torch.sigmoid(ordered_logit_for_route)
                ordered_update = ordered_gate * (ordered_pred - pred)
                if self.use_orthogonal_ordered_patch:
                    # The waveform branch may alter trajectory shape but not
                    # the per-variable horizon level predicted by the anchor
                    # path. This is an exact decoder-space projection.
                    ordered_update = ordered_update - ordered_update.mean(dim=1, keepdim=True)
                pred = pred + ordered_update
        trajectory_gate = None
        if self.use_lowrank_trajectory:
            trajectory_pred = self._lowrank_trajectory_forecast(h)
            if self.use_lowrank_trajectory_only:
                pred = trajectory_pred
            else:
                if self.trajectory_gate is None:
                    raise RuntimeError("low-rank trajectory gate is unavailable")
                trajectory_gate = torch.sigmoid(
                    self.trajectory_gate(torch.cat([last_stat, enc], dim=-1))
                )
                trajectory_gate = trajectory_gate.permute(0, 2, 1)
                pred = pred + trajectory_gate * (trajectory_pred - pred)
        if anchor_mean is not None and anchor_scale is not None and not self.disable_anchor_restore:
            pred = self._restore_anchor(pred, anchor_mean, anchor_scale)
        if self.use_pose_canonical:
            pred = self._restore_pose(pred)
        f_dim = -1 if self.configs.features == "MS" else 0
        out = {"pred": pred[:, -y.shape[1]:, f_dim:], "true": y[:, :, f_dim:], "mask": y_mask[:, :, f_dim:]}
        if (
            self.training
            and self.use_risk_supervised_ordered_patch
            and ordered_pred_for_route is not None
            and ordered_logit_for_route is not None
        ):
            base_route = base_pred_for_route
            ordered_route = ordered_pred_for_route
            if anchor_mean is not None and anchor_scale is not None and not self.disable_anchor_restore:
                base_route = self._restore_anchor(base_route, anchor_mean, anchor_scale)
                ordered_route = self._restore_anchor(ordered_route, anchor_mean, anchor_scale)
            if self.use_pose_canonical:
                base_route = self._restore_pose(base_route)
                ordered_route = self._restore_pose(ordered_route)
            base_route = base_route[:, -y.shape[1]:, f_dim:]
            ordered_route = ordered_route[:, -y.shape[1]:, f_dim:]
            route_logit = ordered_logit_for_route[:, -y.shape[1]:, f_dim:]
            route_mask = out["mask"].to(dtype=pred.dtype)
            base_error = (base_route - out["true"]).square() + 0.5 * (base_route - out["true"]).abs()
            ordered_error = (ordered_route - out["true"]).square() + 0.5 * (ordered_route - out["true"]).abs()
            preference = torch.sigmoid((base_error - ordered_error) / 0.1).detach()
            route_loss = F.binary_cross_entropy_with_logits(
                route_logit,
                preference,
                reduction="none",
            )
            route_loss = (route_loss * route_mask).sum() / route_mask.sum().clamp_min(1.0)
            expert_loss = (ordered_error * route_mask).sum() / route_mask.sum().clamp_min(1.0)
            out["aux_loss"] = 0.05 * route_loss + 0.10 * expert_loss
        if self._pose_frame_usage is not None:
            out["pose_frame_usage"] = self._pose_frame_usage.detach()
        if waveform_gate is not None:
            out["waveform_gate"] = waveform_gate.detach()
        if waveform_cross_gate is not None:
            out["waveform_cross_gate"] = waveform_cross_gate.detach()
        if ordered_gate is not None:
            out["ordered_patch_gate"] = ordered_gate.detach()
        if hpq_gate is not None:
            out["horizon_patch_gate"] = hpq_gate.detach()
        if trajectory_gate is not None:
            out["lowrank_trajectory_gate"] = trajectory_gate.detach()
        if spectral_gate is not None:
            out["spectral_trajectory_gate"] = spectral_gate.detach()
        if spectral_weight is not None:
            out["spectral_trajectory_weight"] = spectral_weight.detach()
        subject_aux = self._subject_robust_aux(
            out["pred"],
            out["true"],
            out["mask"],
            kwargs.get("subject_ID"),
            kwargs.get("subject_weight"),
        )
        if subject_aux is not None:
            out["aux_loss"] = out.get("aux_loss", 0.0) + subject_aux
        out.update(
            {
                "mesa_admit": admit.detach(),
                "mesa_density": stat[..., 0].detach(),
                "mesa_obs": stat[..., 1].detach(),
                "mesa_recency": stat[..., 10].detach(),
                "mesa_span": stat[..., 9].detach(),
                "mesa_value_var": stat[..., 3].detach(),
                "mesa_coobs": stat[..., 1].detach(),
            }
        )
        if self._cag_scale_alpha is not None:
            out["mesa_scale_alpha"] = self._cag_scale_alpha[:, :, :, :n_patch].detach()
        scale_width = self.ct_log_width.exp().to(device=pred.device, dtype=pred.dtype)
        out["mesa_scale_width"] = scale_width.unsqueeze(0).expand(batch, -1).detach()
        if self.training and "no_prob" not in self.mode:
            sigma = torch.nn.functional.softplus(self.ct_sigma_decoder(dec_in).squeeze(-1).permute(0, 2, 1)) + 1e-3
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
