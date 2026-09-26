import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


def _time_default(length: int, batch_size: int, device, dtype) -> Tensor:
    denom = max(length - 1, 1)
    t = torch.arange(length, dtype=dtype, device=device) / denom
    return t.view(1, length, 1).expand(batch_size, -1, -1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Model(nn.Module):
    """Mechanism-Conditioned Predictive State Network.

    MPS-Net is a native forecasting backbone. It maintains variable-level
    predictive states and a global wearable state, then updates them through
    time-flow, silence, arrival, and co-arrival operators before decoding future
    queries. It does not use a base forecaster or residual correction path.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.n_vars = int(configs.enc_in)
        self.d_model = max(8, int(configs.d_model))
        self.hidden = max(self.d_model, int(configs.d_ff // 8) if configs.d_ff else self.d_model)
        self.dropout = float(configs.dropout)
        self.mode = str(getattr(configs, "ablation_name", "") or "").lower()
        self.aux_weight = 0.0 if "no_aux" in self.mode else 0.05
        self.smooth_weight = 1e-4

        self.var_emb = nn.Parameter(torch.randn(self.n_vars, self.d_model) * 0.02)
        self.init_proj = nn.Linear(self.d_model + 7, self.d_model)

        flow_in = self.d_model * 3 + 8
        self.time_flow = MLP(flow_in, self.hidden, self.d_model * 2, self.dropout)
        self.global_flow = MLP(self.d_model * 2 + 1, self.hidden, self.d_model * 2, self.dropout)

        silence_in = self.d_model * 3 + 8
        self.silence_gate = MLP(silence_in, self.hidden, self.d_model, self.dropout)
        self.silence_state = MLP(silence_in, self.hidden, self.d_model, self.dropout)

        event_in = self.d_model * 3 + 11
        self.event_proj = MLP(event_in, self.hidden, self.d_model, self.dropout)
        self.arrival_gru = nn.GRUCell(self.d_model, self.d_model)

        co_in = self.d_model * 2 + 7
        self.co_score = MLP(co_in, self.hidden, 1, self.dropout)
        self.co_value = nn.Linear(self.d_model, self.d_model)
        self.global_gru = nn.GRUCell(self.d_model, self.d_model)
        self.feedback_gate = MLP(co_in + self.d_model, self.hidden, self.d_model, self.dropout)
        self.feedback_proj = nn.Linear(self.d_model, self.d_model)

        dec_in = self.d_model * 3 + 8
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.future_mean_head = nn.Linear(self.d_model, 1)
        self.future_obs_head = nn.Linear(self.d_model, 1)

        self.latest_diag = {}

    def _time(self, mark: Tensor, x: Tensor) -> Tensor:
        if mark is None:
            return _time_default(x.size(1), x.size(0), x.device, x.dtype)
        if mark.dim() == 2:
            mark = mark.unsqueeze(-1)
        return torch.nan_to_num(mark[:, :, [0]].to(dtype=x.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _mechanism_features(
        self,
        t: Tensor,
        obs: Tensor,
        x_t: Tensor,
        last_t: Tensor,
        last_x: Tensor,
        count: Tensor,
        ema_density: Tensor,
        last_gap: Tensor,
        step: int,
        seq_len: int,
    ) -> Tensor:
        dt_last = (t - last_t).clamp_min(0.0)
        count_norm = torch.log1p(count) / math.log1p(max(seq_len, 1))
        recent_density = ema_density
        gap = dt_last
        gap_delta = (gap - last_gap).abs()
        trend = x_t - last_x
        co_obs = obs.mean(dim=1, keepdim=True).expand_as(obs)
        step_frac = obs.new_full(obs.shape, float(step) / max(seq_len - 1, 1))
        return torch.stack(
            [
                dt_last,
                count_norm,
                recent_density,
                gap,
                gap_delta,
                last_x,
                trend,
            ],
            dim=-1,
        ), co_obs, step_frac

    def _flow(self, p: Tensor, g: Tensor, emb: Tensor, mech: Tensor, dt: Tensor) -> Tensor:
        batch_size, n_vars, _ = p.shape
        g_expand = g.unsqueeze(1).expand(-1, n_vars, -1)
        dt_expand = dt.view(batch_size, 1, 1).expand(-1, n_vars, 1)
        inp = torch.cat([p, g_expand, emb, mech, dt_expand], dim=-1)
        params = self.time_flow(inp)
        decay_raw, target = params.chunk(2, dim=-1)
        decay = torch.exp(-F.softplus(decay_raw) * dt_expand).clamp(0.0, 1.0)
        return decay * p + (1.0 - decay) * torch.tanh(target)

    def _global_flow(self, g: Tensor, pooled: Tensor, dt: Tensor) -> Tensor:
        params = self.global_flow(torch.cat([g, pooled, dt.view(-1, 1)], dim=-1))
        decay_raw, target = params.chunk(2, dim=-1)
        decay = torch.exp(-F.softplus(decay_raw) * dt.view(-1, 1)).clamp(0.0, 1.0)
        return decay * g + (1.0 - decay) * torch.tanh(target)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor = None,
        x_mask: Tensor = None,
        y: Tensor = None,
        y_mark: Tensor = None,
        y_mask: Tensor = None,
        exp_stage: str = "train",
        **kwargs,
    ) -> dict:
        batch_size, seq_len, n_vars = x.shape
        if n_vars != self.n_vars:
            raise ValueError(f"MPSNet expected enc_in={self.n_vars}, got x.shape[-1]={n_vars}")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_timeline = self._time(x_mark, x)
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x_mask = torch.nan_to_num(x_mask.to(dtype=x.dtype), nan=0.0).clamp(0.0, 1.0)
        if y is None:
            y = torch.zeros(batch_size, self.pred_len, n_vars, dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y)
        if y_mark is None:
            y_mark = _time_default(y.shape[1], batch_size, x.device, x.dtype)
        else:
            y_mark = self._time(y_mark, y)

        emb = self.var_emb.to(device=x.device, dtype=x.dtype).view(1, n_vars, self.d_model).expand(batch_size, -1, -1)
        zero_mech = x.new_zeros(batch_size, n_vars, 7)
        p = torch.tanh(self.init_proj(torch.cat([emb, zero_mech], dim=-1)))
        g = p.mean(dim=1)

        last_t = x.new_zeros(batch_size, n_vars)
        last_x = x.new_zeros(batch_size, n_vars)
        last_gap = x.new_zeros(batch_size, n_vars)
        count = x.new_zeros(batch_size, n_vars)
        ema_density = x.new_zeros(batch_size, n_vars)
        prev_t = x_timeline[:, 0, 0]
        smooth_terms = []
        silence_activity = []
        coarrival_mass = []

        silence_mask_source = x_mask
        if "shuffled_silence" in self.mode and batch_size > 1:
            silence_mask_source = torch.roll(x_mask, shifts=1, dims=0)

        for step in range(seq_len):
            t = x_timeline[:, step, 0]
            dt = (t - prev_t).clamp_min(0.0)
            obs = x_mask[:, step, :]
            silence_obs = silence_mask_source[:, step, :]
            x_step = x[:, step, :]
            ema_density = 0.95 * ema_density + 0.05 * obs
            mech, co_obs, step_frac = self._mechanism_features(
                t.view(batch_size, 1).expand(-1, n_vars),
                obs,
                x_step,
                last_t,
                last_x,
                count,
                ema_density,
                last_gap,
                step,
                seq_len,
            )
            mech_for_flow = mech
            pooled = p.mean(dim=1)
            p_prev = p
            p = self._flow(p, g, emb, mech_for_flow, dt)
            g = self._global_flow(g, pooled, dt)

            if "no_silence" not in self.mode and "arrival_only" not in self.mode:
                silence_in = torch.cat([p, g.unsqueeze(1).expand(-1, n_vars, -1), emb, mech, dt.view(batch_size, 1, 1).expand(-1, n_vars, 1)], dim=-1)
                s_gate = torch.sigmoid(self.silence_gate(silence_in))
                s_state = torch.tanh(self.silence_state(silence_in))
                silent = (1.0 - silence_obs).unsqueeze(-1)
                p = silent * ((1.0 - s_gate) * p + s_gate * s_state) + (1.0 - silent) * p
                silence_activity.append((silent * s_gate).mean().detach())

            event_in = torch.cat(
                [
                    p,
                    g.unsqueeze(1).expand(-1, n_vars, -1),
                    emb,
                    mech,
                    co_obs.unsqueeze(-1),
                    step_frac.unsqueeze(-1),
                ],
                dim=-1,
            )
            event_token = torch.tanh(self.event_proj(torch.cat([event_in, x_step.unsqueeze(-1), obs.unsqueeze(-1)], dim=-1)))
            p_flat = p.reshape(batch_size * n_vars, self.d_model)
            e_flat = event_token.reshape(batch_size * n_vars, self.d_model)
            arrived = self.arrival_gru(e_flat, p_flat).view(batch_size, n_vars, self.d_model)
            p = obs.unsqueeze(-1) * arrived + (1.0 - obs).unsqueeze(-1) * p

            if "no_coarrival" not in self.mode and "arrival_only" not in self.mode:
                co_in = torch.cat([p, emb, mech], dim=-1)
                score = self.co_score(co_in).squeeze(-1)
                score = score.masked_fill(obs <= 0.0, -1e4)
                has_obs = (obs.sum(dim=1, keepdim=True) > 0).to(dtype=x.dtype)
                attn = torch.softmax(score, dim=1) * has_obs
                obs_sum = attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
                attn = attn / obs_sum
                co_msg = (attn.unsqueeze(-1) * self.co_value(p)).sum(dim=1)
                g = self.global_gru(co_msg, g)
                fb_in = torch.cat([p, emb, mech, g.unsqueeze(1).expand(-1, n_vars, -1)], dim=-1)
                fb_gate = torch.sigmoid(self.feedback_gate(fb_in))
                p = p + fb_gate * self.feedback_proj(g).unsqueeze(1)
                coarrival_mass.append(obs.mean().detach())

            smooth_terms.append((p - p_prev).pow(2).mean())
            observed = obs > 0.0
            new_gap = (t.view(batch_size, 1) - last_t).clamp_min(0.0)
            last_gap = torch.where(observed, new_gap, last_gap)
            last_t = torch.where(observed, t.view(batch_size, 1).expand(-1, n_vars), last_t)
            last_x = torch.where(observed, x_step, last_x)
            count = count + obs
            prev_t = t

        pred_len = y.shape[1]
        y_time = self._time(y_mark, y)[:, :, 0]
        tau = (y_time - prev_t.view(batch_size, 1)).clamp_min(0.0)
        p_exp = p.unsqueeze(2).expand(-1, -1, pred_len, -1)
        g_exp = g.view(batch_size, 1, 1, self.d_model).expand(-1, n_vars, pred_len, -1)
        emb_exp = emb.unsqueeze(2).expand(-1, -1, pred_len, -1)
        tau_exp = tau.view(batch_size, 1, pred_len, 1).expand(-1, n_vars, -1, 1)
        last_dt = tau_exp + (prev_t.view(batch_size, 1, 1, 1) - last_t.view(batch_size, n_vars, 1, 1)).clamp_min(0.0)
        count_norm = (torch.log1p(count) / math.log1p(max(seq_len, 1))).view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1)
        dec_mech = torch.cat(
            [
                tau_exp,
                last_dt,
                count_norm,
                ema_density.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                last_x.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                last_gap.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                torch.zeros_like(tau_exp),
                torch.ones_like(tau_exp),
            ],
            dim=-1,
        )
        dec_in = torch.cat([p_exp, g_exp, emb_exp, dec_mech], dim=-1)
        pred = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)

        f_dim = -1 if self.configs.features == "MS" else 0
        outputs = {
            "pred": pred[:, -y.shape[1] :, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:],
        }

        aux_terms = []
        if self.training and self.aux_weight > 0.0 and y is not None:
            ym = y_mask.to(dtype=y.dtype)
            denom = ym.sum(dim=1).clamp_min(1.0)
            future_mean = (y * ym).sum(dim=1) / denom
            future_obs = ym.mean(dim=1).clamp(0.0, 1.0)
            mean_pred = self.future_mean_head(p).squeeze(-1)
            obs_logit = self.future_obs_head(p).squeeze(-1)
            aux_terms.append(F.l1_loss(mean_pred, future_mean))
            aux_terms.append(F.binary_cross_entropy_with_logits(obs_logit, future_obs))
        if self.training and self.smooth_weight > 0.0 and smooth_terms:
            aux_terms.append(torch.stack(smooth_terms).mean() * self.smooth_weight)
        if aux_terms:
            outputs["aux_loss"] = self.aux_weight * sum(aux_terms[:-1]) + aux_terms[-1]

        self.latest_diag = {
            "silence_gate_mean": float(torch.stack(silence_activity).mean().detach().cpu()) if silence_activity else 0.0,
            "coarrival_mass": float(torch.stack(coarrival_mass).mean().detach().cpu()) if coarrival_mass else 0.0,
        }
        return outputs


