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
    """Arrival-State Network.

    ASN-Net is a native predictive-state backbone distilled from v412: explicit
    silence transitions were noisy on OPPORTUNITY, while arrival-driven state
    evolution was useful. ASN-Net keeps time flow, arrival updates, and light
    co-arrival coupling, with no residual branch and no carrier model.
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
        self.aux_weight = 0.0 if "no_aux" in self.mode else 0.03
        self.smooth_weight = 5e-5

        self.var_emb = nn.Parameter(torch.randn(self.n_vars, self.d_model) * 0.02)
        self.init_proj = nn.Linear(self.d_model + 8, self.d_model)
        self.state_norm = nn.LayerNorm(self.d_model)
        self.global_norm = nn.LayerNorm(self.d_model)

        self.flow = MLP(self.d_model * 3 + 9, self.hidden, self.d_model * 2, self.dropout)
        self.global_flow = MLP(self.d_model * 2 + 1, self.hidden, self.d_model * 2, self.dropout)
        self.event_token = MLP(self.d_model * 3 + 10, self.hidden, self.d_model, self.dropout)
        self.arrival_gru = nn.GRUCell(self.d_model, self.d_model)
        self.co_score = MLP(self.d_model * 2 + 8, self.hidden, 1, self.dropout)
        self.co_value = nn.Linear(self.d_model, self.d_model)
        self.global_gru = nn.GRUCell(self.d_model, self.d_model)
        self.co_admit = MLP(self.d_model + 3, self.hidden, 1, self.dropout)
        self.feedback_gate = MLP(self.d_model * 3 + 8, self.hidden, self.d_model, self.dropout)
        self.feedback_value = nn.Linear(self.d_model, self.d_model)
        nn.init.constant_(self.co_admit.net[-1].bias, -3.0)
        nn.init.constant_(self.feedback_gate.net[-1].bias, -2.0)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model * 3 + 8, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.future_mean = nn.Linear(self.d_model, 1)
        self.future_obs = nn.Linear(self.d_model, 1)

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
        step: int,
        seq_len: int,
    ) -> Tensor:
        dt_last = (t - last_t).clamp_min(0.0)
        count_norm = torch.log1p(count) / math.log1p(max(seq_len, 1))
        trend = x_t - last_x
        co_obs = obs.mean(dim=1, keepdim=True).expand_as(obs)
        step_frac = obs.new_full(obs.shape, float(step) / max(seq_len - 1, 1))
        return torch.stack(
            [dt_last, count_norm, ema_density, last_x, trend, last_gap, co_obs, step_frac],
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
        p = self.state_norm(self.init_proj(torch.cat([emb, x.new_zeros(batch_size, n_vars, 8)], dim=-1)))
        g = self.global_norm(p.mean(dim=1))
        last_t = x.new_zeros(batch_size, n_vars)
        last_x = x.new_zeros(batch_size, n_vars)
        last_gap = x.new_zeros(batch_size, n_vars)
        count = x.new_zeros(batch_size, n_vars)
        ema_density = x.new_zeros(batch_size, n_vars)
        prev_t = x_time[:, 0, 0]
        smooth_terms = []

        for step in range(seq_len):
            t = x_time[:, step, 0]
            dt = (t - prev_t).clamp_min(0.0)
            obs = x_mask[:, step, :]
            x_step = x[:, step, :]
            ema_density = 0.96 * ema_density + 0.04 * obs
            mech = self._mech(
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

            if "no_coarrival" not in self.mode:
                co_in = torch.cat([p, emb, mech], dim=-1)
                score = self.co_score(co_in).squeeze(-1).masked_fill(obs <= 0.0, -1e4)
                has_obs = (obs.sum(dim=1, keepdim=True) > 0).to(dtype=x.dtype)
                attn = torch.softmax(score, dim=1) * has_obs
                attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
                msg = (attn.unsqueeze(-1) * self.co_value(p)).sum(dim=1)
                g_next = self.global_norm(self.global_gru(msg, g))
                if "constrained_coarrival" in self.mode or "sparse_coarrival" in self.mode or "tiny_coarrival" in self.mode:
                    co_cap = 0.05 if "tiny_coarrival" in self.mode else (0.10 if "sparse_coarrival" in self.mode else 0.25)
                    obs_frac = obs.mean(dim=1, keepdim=True)
                    obs_any = (obs.sum(dim=1, keepdim=True) > 0).to(dtype=x.dtype)
                    obs_var = obs.var(dim=1, keepdim=True, unbiased=False)
                    admit_in = torch.cat([msg, obs_frac, obs_any, obs_var], dim=-1)
                    admit = co_cap * torch.sigmoid(self.co_admit(admit_in))
                    g = self.global_norm(g + admit * (g_next - g))
                else:
                    g = g_next
                if "no_feedback" not in self.mode:
                    fb_in = torch.cat([p, emb, mech, g.unsqueeze(1).expand(-1, n_vars, -1)], dim=-1)
                    gate = torch.sigmoid(self.feedback_gate(fb_in))
                    if "constrained_coarrival" in self.mode or "sparse_coarrival" in self.mode or "tiny_coarrival" in self.mode:
                        co_cap = 0.05 if "tiny_coarrival" in self.mode else (0.10 if "sparse_coarrival" in self.mode else 0.25)
                        gate = co_cap * gate
                    p = self.state_norm(p + gate * self.feedback_value(g).unsqueeze(1))

            smooth_terms.append((p - p_prev).pow(2).mean())
            observed = obs > 0.0
            new_gap = (t.view(batch_size, 1) - last_t).clamp_min(0.0)
            last_gap = torch.where(observed, new_gap, last_gap)
            last_t = torch.where(observed, t.view(batch_size, 1).expand(-1, n_vars), last_t)
            last_x = torch.where(observed, x_step, last_x)
            count = count + obs
            prev_t = t

        pred_len = y.shape[1]
        yt = y_time[:, :, 0]
        tau = (yt - prev_t.view(batch_size, 1)).clamp_min(0.0)
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
        pred = self.decoder(torch.cat([p_exp, g_exp, emb_exp, dec_mech], dim=-1)).squeeze(-1).permute(0, 2, 1)
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
            aux_terms.append(F.l1_loss(self.future_mean(p).squeeze(-1), future_mean))
            aux_terms.append(F.binary_cross_entropy_with_logits(self.future_obs(p).squeeze(-1), future_obs))
        if self.training and smooth_terms:
            aux_terms.append(torch.stack(smooth_terms).mean() * self.smooth_weight)
        if aux_terms:
            outputs["aux_loss"] = self.aux_weight * sum(aux_terms[:-1]) + aux_terms[-1]
        return outputs

