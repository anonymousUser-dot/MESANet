import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.ExpConfigs import ExpConfigs


def _default_time(length: int, batch_size: int, device, dtype) -> Tensor:
    denom = max(length - 1, 1)
    t = torch.arange(length, device=device, dtype=dtype) / denom
    return t.view(1, length, 1).expand(batch_size, -1, -1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
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
    """Arrival Local-Integral Network.

    ALINet is a native IMTS forecasting backbone. It maintains per-variable
    arrival states and decodes futures from two mechanism-conditioned operators:
    a local operator for the last observed state and an integral operator for
    interval evidence. The final forecast is produced directly by a closed
    mechanism gate; there is no base forecaster, residual branch, or post-hoc
    correction path.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = getattr(configs, "pred_len_max_irr", None) or configs.pred_len
        self.n_vars = int(configs.enc_in)
        self.d_model = max(8, int(configs.d_model))
        self.hidden = max(self.d_model * 2, int(getattr(configs, "d_ff", self.d_model * 4) // 4))
        self.dropout = float(configs.dropout)
        self.mode = str(getattr(configs, "ablation_name", "") or "").lower()
        self.aux_weight = 0.0 if "no_aux" in self.mode else 0.02
        self.smooth_weight = 5e-5

        self.var_emb = nn.Parameter(torch.randn(self.n_vars, self.d_model) * 0.02)
        self.init_proj = nn.Linear(self.d_model + 10, self.d_model)
        self.state_norm = nn.LayerNorm(self.d_model)
        self.global_norm = nn.LayerNorm(self.d_model)
        self.integral_norm = nn.LayerNorm(self.d_model)

        self.flow = MLP(self.d_model * 3 + 11, self.hidden, self.d_model * 2, self.dropout)
        self.global_flow = MLP(self.d_model * 2 + 1, self.hidden, self.d_model * 2, self.dropout)
        self.event_token = MLP(self.d_model * 3 + 12, self.hidden, self.d_model, self.dropout)
        self.arrival_gru = nn.GRUCell(self.d_model, self.d_model)

        self.integral_proj = MLP(self.d_model * 2 + 10, self.hidden, self.d_model, self.dropout)
        dec_in = self.d_model * 3 + 10
        self.local_decoder = nn.Sequential(
            nn.Linear(dec_in, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.integral_decoder = nn.Sequential(
            nn.Linear(dec_in, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.gate = MLP(self.d_model * 3 + 10, self.hidden, 1, self.dropout)
        nn.init.constant_(self.gate.net[-1].bias, -1.0)

        self.future_mean = nn.Linear(self.d_model * 2, 1)
        self.future_obs = nn.Linear(self.d_model * 2, 1)
        self.latest_diag = {}

    def _time(self, mark: Tensor | None, ref: Tensor) -> Tensor:
        if mark is None:
            return _default_time(ref.size(1), ref.size(0), ref.device, ref.dtype)
        if mark.dim() == 2:
            mark = mark.unsqueeze(-1)
        return torch.nan_to_num(mark[:, :, [0]].to(dtype=ref.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _mech(
        self,
        t: Tensor,
        obs: Tensor,
        x_t: Tensor,
        last_t: Tensor,
        last_x: Tensor,
        count: Tensor,
        ema_density: Tensor,
        last_gap: Tensor,
        mean_x: Tensor,
        second_x: Tensor,
        first_t: Tensor,
        step: int,
        seq_len: int,
    ) -> Tensor:
        dt_last = (t - last_t).clamp_min(0.0)
        count_norm = torch.log1p(count) / math.log1p(max(seq_len, 1))
        trend = x_t - last_x
        var_x = (second_x - mean_x.pow(2)).clamp_min(0.0)
        span = (t - first_t).clamp_min(0.0)
        co_obs = obs.mean(dim=1, keepdim=True).expand_as(obs)
        step_frac = obs.new_full(obs.shape, float(step) / max(seq_len - 1, 1))
        return torch.stack(
            [dt_last, count_norm, ema_density, last_x, trend, last_gap, mean_x, var_x, span, co_obs, step_frac],
            dim=-1,
        )

    def _flow(self, p: Tensor, g: Tensor, emb: Tensor, mech: Tensor, dt: Tensor) -> Tensor:
        batch_size, n_vars, _ = p.shape
        dtv = dt.view(batch_size, 1, 1).expand(-1, n_vars, 1)
        inp = torch.cat([p, g.unsqueeze(1).expand(-1, n_vars, -1), emb, mech, dtv], dim=-1)
        decay_raw, target = self.flow(inp).chunk(2, dim=-1)
        decay = torch.exp(-F.softplus(decay_raw) * dtv).clamp(0.0, 1.0)
        return self.state_norm(decay * p + (1.0 - decay) * torch.tanh(target))

    def _global_flow(self, g: Tensor, pooled: Tensor, dt: Tensor) -> Tensor:
        decay_raw, target = self.global_flow(torch.cat([g, pooled, dt.view(-1, 1)], dim=-1)).chunk(2, dim=-1)
        decay = torch.exp(-F.softplus(decay_raw) * dt.view(-1, 1)).clamp(0.0, 1.0)
        return self.global_norm(decay * g + (1.0 - decay) * torch.tanh(target))

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
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_time = self._time(x_mark, x)
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x_mask = torch.nan_to_num(x_mask.to(dtype=x.dtype), nan=0.0).clamp(0.0, 1.0)
        if y is None:
            y = torch.zeros(batch_size, self.pred_len, n_vars, dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y)
        y_time = self._time(y_mark, y) if y_mark is not None else _default_time(y.shape[1], batch_size, x.device, x.dtype)

        emb = self.var_emb.to(device=x.device, dtype=x.dtype).view(1, n_vars, self.d_model).expand(batch_size, -1, -1)
        zero_mech = x.new_zeros(batch_size, n_vars, 10)
        p = self.state_norm(self.init_proj(torch.cat([emb, zero_mech], dim=-1)))
        g = self.global_norm(p.mean(dim=1))

        last_t = x.new_zeros(batch_size, n_vars)
        first_t = x.new_zeros(batch_size, n_vars)
        seen = x.new_zeros(batch_size, n_vars)
        last_x = x.new_zeros(batch_size, n_vars)
        last_gap = x.new_zeros(batch_size, n_vars)
        count = x.new_zeros(batch_size, n_vars)
        ema_density = x.new_zeros(batch_size, n_vars)
        int_sum = x.new_zeros(batch_size, n_vars)
        int_sq_sum = x.new_zeros(batch_size, n_vars)
        int_ema = x.new_zeros(batch_size, n_vars)
        int_ema_seen = x.new_zeros(batch_size, n_vars)
        prev_t = x_time[:, 0, 0]
        smooth_terms = []
        gate_terms = []

        for step in range(seq_len):
            t = x_time[:, step, 0]
            dt = (t - prev_t).clamp_min(0.0)
            obs = x_mask[:, step, :]
            x_step = x[:, step, :]
            ema_density = 0.96 * ema_density + 0.04 * obs
            mean_x = int_sum / count.clamp_min(1.0)
            second_x = int_sq_sum / count.clamp_min(1.0)
            mech = self._mech(
                t.view(batch_size, 1).expand(-1, n_vars),
                obs,
                x_step,
                last_t,
                last_x,
                count,
                ema_density,
                last_gap,
                mean_x,
                second_x,
                first_t,
                step,
                seq_len,
            )
            p_prev = p
            p = self._flow(p, g, emb, mech, dt)
            g = self._global_flow(g, p_prev.mean(dim=1), dt)

            event_in = torch.cat(
                [p, g.unsqueeze(1).expand(-1, n_vars, -1), emb, mech, x_step.unsqueeze(-1), obs.unsqueeze(-1)],
                dim=-1,
            )
            token = torch.tanh(self.event_token(event_in))
            p_arrive = self.arrival_gru(
                token.reshape(batch_size * n_vars, self.d_model),
                p.reshape(batch_size * n_vars, self.d_model),
            ).view(batch_size, n_vars, self.d_model)
            p = self.state_norm(obs.unsqueeze(-1) * p_arrive + (1.0 - obs).unsqueeze(-1) * p)

            observed = obs > 0.0
            new_gap = (t.view(batch_size, 1) - last_t).clamp_min(0.0)
            first_t = torch.where((seen <= 0.0) & observed, t.view(batch_size, 1).expand(-1, n_vars), first_t)
            seen = torch.maximum(seen, obs)
            last_gap = torch.where(observed, new_gap, last_gap)
            last_t = torch.where(observed, t.view(batch_size, 1).expand(-1, n_vars), last_t)
            last_x = torch.where(observed, x_step, last_x)
            count = count + obs
            int_sum = int_sum + obs * x_step
            int_sq_sum = int_sq_sum + obs * x_step.pow(2)
            # Exponential interval summary: stable under sparse observations and still deployment-valid.
            int_ema = 0.95 * int_ema + 0.05 * obs * x_step
            int_ema_seen = 0.95 * int_ema_seen + 0.05 * obs
            smooth_terms.append((p - p_prev).pow(2).mean())
            prev_t = t

        pred_len = y.shape[1]
        yt = y_time[:, :, 0]
        tau = (yt - prev_t.view(batch_size, 1)).clamp_min(0.0)
        mean_x = int_sum / count.clamp_min(1.0)
        second_x = int_sq_sum / count.clamp_min(1.0)
        var_x = (second_x - mean_x.pow(2)).clamp_min(0.0)
        span = (prev_t.view(batch_size, 1) - first_t).clamp_min(0.0)
        ema_mean = int_ema / int_ema_seen.clamp_min(1e-4)
        count_norm = torch.log1p(count) / math.log1p(max(seq_len, 1))

        int_mech = torch.stack(
            [
                count_norm,
                ema_density,
                mean_x,
                var_x,
                ema_mean,
                last_x,
                last_gap,
                span,
                (prev_t.view(batch_size, 1) - last_t).clamp_min(0.0),
                seen,
            ],
            dim=-1,
        )
        z_int = self.integral_norm(self.integral_proj(torch.cat([p, emb, int_mech], dim=-1)))

        p_exp = p.unsqueeze(2).expand(-1, -1, pred_len, -1)
        z_exp = z_int.unsqueeze(2).expand(-1, -1, pred_len, -1)
        g_exp = g.view(batch_size, 1, 1, self.d_model).expand(-1, n_vars, pred_len, -1)
        emb_exp = emb.unsqueeze(2).expand(-1, -1, pred_len, -1)
        tau_exp = tau.view(batch_size, 1, pred_len, 1).expand(-1, n_vars, -1, 1)
        last_dt = tau_exp + (prev_t.view(batch_size, 1, 1, 1) - last_t.view(batch_size, n_vars, 1, 1)).clamp_min(0.0)
        dec_mech = torch.cat(
            [
                tau_exp,
                last_dt,
                count_norm.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                ema_density.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                last_x.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                last_gap.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                mean_x.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                var_x.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                span.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
                seen.view(batch_size, n_vars, 1, 1).expand(-1, -1, pred_len, -1),
            ],
            dim=-1,
        )
        local_in = torch.cat([p_exp, g_exp, emb_exp, dec_mech], dim=-1)
        integral_in = torch.cat([z_exp, g_exp, emb_exp, dec_mech], dim=-1)
        y_local = self.local_decoder(local_in).squeeze(-1).permute(0, 2, 1)
        y_integral = self.integral_decoder(integral_in).squeeze(-1).permute(0, 2, 1)
        gate_logits = self.gate(torch.cat([p_exp, z_exp, emb_exp, dec_mech], dim=-1)).squeeze(-1).permute(0, 2, 1)
        gate = torch.sigmoid(gate_logits)
        if "local_only" in self.mode:
            pred = y_local
            gate = torch.zeros_like(gate)
        elif "integral_only" in self.mode:
            pred = y_integral
            gate = torch.ones_like(gate)
        elif "no_mechanism_gate" in self.mode or "mean_gate" in self.mode:
            gate = torch.full_like(gate, 0.5)
            pred = 0.5 * y_local + 0.5 * y_integral
        else:
            pred = (1.0 - gate) * y_local + gate * y_integral
        gate_terms.append(gate.detach().mean())

        f_dim = -1 if self.configs.features == "MS" else 0
        outputs = {
            "pred": pred[:, -y.shape[1] :, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:],
        }

        aux_terms = []
        if self.training and self.aux_weight > 0.0:
            ym = y_mask.to(dtype=y.dtype)
            denom = ym.sum(dim=1).clamp_min(1.0)
            future_mean = (y * ym).sum(dim=1) / denom
            future_obs = ym.mean(dim=1).clamp(0.0, 1.0)
            aux_state = torch.cat([p, z_int], dim=-1)
            aux_terms.append(F.l1_loss(self.future_mean(aux_state).squeeze(-1), future_mean))
            aux_terms.append(F.binary_cross_entropy_with_logits(self.future_obs(aux_state).squeeze(-1), future_obs))
        if self.training and smooth_terms:
            aux_terms.append(torch.stack(smooth_terms).mean() * self.smooth_weight)
        if aux_terms:
            outputs["aux_loss"] = self.aux_weight * sum(aux_terms[:-1]) + aux_terms[-1]
        self.latest_diag = {
            "local_integral_gate_mean": float(torch.stack(gate_terms).mean().detach().cpu()) if gate_terms else 0.0,
            "observed_density": float((count / max(seq_len, 1)).mean().detach().cpu()),
        }
        return outputs
