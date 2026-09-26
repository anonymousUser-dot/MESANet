import math

import torch
import torch.nn as nn
from torch import Tensor

from models.AMPGNet import FourierTime, MLP, _time
from utils.ExpConfigs import ExpConfigs


class Model(nn.Module):
    """Causal Event-Field Operator Network.

    CEFONet is a first-principles IMTS backbone. It treats the input as an
    irregular observation measure rather than a pre-cut patch sequence. A
    causal multiscale event-field operator maps observed events to latent query
    states; temporal and variable operators then mix those states before a
    future decoder predicts the target sequence.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.n_vars = int(configs.enc_in)
        self.d_model = max(32, int(configs.d_model))
        self.hidden = max(96, int(getattr(configs, "d_ff", self.d_model * 4) // 2))
        self.dropout = float(configs.dropout)
        self.patch_len = max(2, int(configs.patch_len))
        self.mode = str(getattr(configs, "ablation_name", "") or "").lower()
        self.n_scales = 4
        self.stat_dim = 8
        self.var_emb = nn.Parameter(torch.randn(self.n_vars, self.d_model) * 0.02)
        self.log_width = nn.Parameter(torch.log(torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)))
        te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
        self.time_enc = FourierTime(te_dim)
        field_in = self.stat_dim + te_dim + self.d_model + 1
        self.field_op = MLP(field_in, self.hidden, self.d_model, self.dropout)
        self.field_score = MLP(field_in, self.hidden, 1, self.dropout)
        self.field_norm = nn.LayerNorm(self.d_model)
        heads = max(1, min(int(configs.n_heads), self.d_model // 16))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=max(self.hidden, self.d_model * 2),
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=1)
        graph_dim = max(8, int(getattr(configs, "node_dim", 10)))
        self.graph_q = nn.Linear(self.d_model, graph_dim)
        self.graph_k = nn.Linear(self.d_model, graph_dim)
        self.graph_msg = nn.Linear(self.d_model, self.d_model)
        self.graph_gate = MLP(self.d_model + self.stat_dim, self.hidden, self.d_model, self.dropout)
        self.graph_norm = nn.LayerNorm(self.d_model)
        self.pool_score = nn.Linear(self.d_model + self.stat_dim, 1)
        self.future_te = FourierTime(te_dim)
        dec_in = self.d_model * 2 + te_dim + 8
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.latest_diag = {}

    def _get_time(self, mark: Tensor | None, ref: Tensor) -> Tensor:
        if mark is None:
            return _time(ref.size(1), ref.size(0), ref.device, ref.dtype)
        if mark.dim() == 2:
            mark = mark.unsqueeze(-1)
        return torch.nan_to_num(mark[:, :, [0]].to(dtype=ref.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _query_times(self, batch: int, seq_len: int, device, dtype) -> Tensor:
        n_query = max(1, math.ceil(seq_len / self.patch_len))
        q = (torch.arange(n_query, device=device, dtype=dtype) + 1.0) / max(n_query, 1)
        return q.view(1, n_query, 1).expand(batch, -1, -1)

    def _event_field(self, x: Tensor, mask: Tensor, t: Tensor, q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, n_vars = x.shape
        n_query = q.size(1)
        dtype = x.dtype
        device = x.device
        x_bvt = x.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
        m_bvt = mask.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
        t_hist = t[:, :, 0].view(batch, 1, 1, seq_len)
        q_hist = q[:, :, 0].view(batch, 1, n_query, 1)
        dist = q_hist - t_hist
        causal = (dist >= 0).to(dtype)
        base_width = 1.0 / max(n_query, 1)
        widths = (base_width * self.log_width.exp().to(device=device, dtype=dtype)).view(1, self.n_scales, 1, 1)
        if "box_kernel" in self.mode:
            kernel = ((dist >= 0) & (dist <= widths)).to(dtype)
        elif "gaussian_kernel" in self.mode:
            z = dist.clamp_min(0.0) / widths.clamp_min(1.0e-4)
            kernel = torch.exp(-0.5 * z.square().clamp_max(64.0)) * causal
        else:
            kernel = torch.exp(-dist.clamp_min(0.0) / widths.clamp_min(1.0e-4)) * causal

        weight = kernel.unsqueeze(1) * m_bvt
        denom = weight.sum(dim=-1).clamp_min(1.0e-6)
        kernel_mass = kernel.sum(dim=-1).unsqueeze(1).clamp_min(1.0e-6)
        density = (weight.sum(dim=-1) / kernel_mass).clamp(0.0, 1.0)
        mean = (weight * x_bvt).sum(dim=-1) / denom
        second = (weight * x_bvt.square()).sum(dim=-1) / denom
        var = (second - mean.square()).clamp_min(0.0)
        t_exp = t_hist.unsqueeze(1)
        t_mean = (weight * t_exp).sum(dim=-1) / denom
        t_second = (weight * t_exp.square()).sum(dim=-1) / denom
        t_var = (t_second - t_mean.square()).clamp_min(1.0e-6)
        cov = (weight * (x_bvt - mean.unsqueeze(-1)) * (t_exp - t_mean.unsqueeze(-1))).sum(dim=-1) / denom
        slope = cov / t_var.clamp_min(1.0e-4)
        t_std = torch.sqrt(t_var)
        q_scalar = q[:, :, 0].view(batch, 1, 1, n_query).expand(-1, n_vars, self.n_scales, -1)
        recency = (q_scalar - t_mean).clamp_min(0.0)
        stat = torch.stack([density, mean, var, slope, t_mean, t_std, recency, q_scalar], dim=-1)
        stat = torch.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

        te = self.time_enc(q).view(batch, 1, 1, n_query, -1).expand(-1, n_vars, self.n_scales, -1, -1)
        var_emb = self.var_emb.to(device=device, dtype=dtype).view(1, n_vars, 1, 1, self.d_model)
        var_emb = var_emb.expand(batch, -1, self.n_scales, n_query, -1)
        width_feat = self.log_width.to(device=device, dtype=dtype).view(1, 1, self.n_scales, 1, 1)
        width_feat = width_feat.expand(batch, n_vars, -1, n_query, -1)
        base = torch.cat([stat, te, var_emb, width_feat], dim=-1)
        token_k = self.field_op(base)
        if "single_scale" in self.mode:
            alpha = torch.zeros(batch, n_vars, self.n_scales, n_query, 1, device=device, dtype=dtype)
            alpha[:, :, min(1, self.n_scales - 1), :, :] = 1.0
        elif "uniform_scale" in self.mode:
            alpha = torch.full((batch, n_vars, self.n_scales, n_query, 1), 1.0 / self.n_scales, device=device, dtype=dtype)
        else:
            alpha = torch.softmax(self.field_score(base), dim=2)
        token = (alpha * token_k).sum(dim=2)
        fused_stat = (alpha * stat).sum(dim=2)
        has_obs = (denom.max(dim=2).values > 1.0e-6).to(dtype)
        token = self.field_norm(token + self.var_emb.to(device=device, dtype=dtype).view(1, n_vars, 1, self.d_model))
        return token, fused_stat, has_obs

    def _mix(self, token: Tensor, stat: Tensor) -> Tensor:
        h = token
        if "no_temporal" not in self.mode and h.size(2) > 1:
            b, v, q, d = h.shape
            h = self.temporal(h.reshape(b * v, q, d)).view(b, v, q, d)
        if "no_graph" in self.mode:
            return h
        h_q = h.permute(0, 2, 1, 3)
        qv = self.graph_q(h_q)
        kv = self.graph_k(h_q)
        scores = torch.matmul(qv, kv.transpose(-1, -2)) / math.sqrt(max(qv.size(-1), 1))
        attn = torch.softmax(scores, dim=-1)
        msg = torch.matmul(attn, self.graph_msg(h_q))
        msg = msg.permute(0, 2, 1, 3)
        gate = torch.sigmoid(self.graph_gate(torch.cat([h, stat], dim=-1)))
        return self.graph_norm(h + gate * msg)

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
        q = self._query_times(batch, x.size(1), x.device, x.dtype)
        token, stat, has_obs = self._event_field(x, x_mask, t, q)
        h = self._mix(token, stat)
        score = self.pool_score(torch.cat([h, stat], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1.0e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        enc = (weights * h).sum(dim=2)

        last_stat = stat[:, :, -1, :]
        global_state = enc.mean(dim=1, keepdim=True).expand(-1, self.n_vars, -1)
        enc_exp = enc.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        glob_exp = global_state.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y_t = self._get_time(y_mark, y) if y_mark is not None else _time(y.size(1), batch, x.device, x.dtype)
        te_f = self.future_te(y_t).unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        horizon = torch.arange(1, y.size(1) + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        horizon = horizon / max(y.size(1), 1)
        dec_mech = torch.stack(
            [
                horizon.expand(batch, self.n_vars, -1),
                last_stat[..., 0].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 1].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 2].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 3].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 5].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 6].unsqueeze(-1).expand(-1, -1, y.size(1)),
                last_stat[..., 7].unsqueeze(-1).expand(-1, -1, y.size(1)),
            ],
            dim=-1,
        )
        dec_in = torch.cat([enc_exp, glob_exp, te_f, dec_mech], dim=-1)
        pred = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)
        f_dim = -1 if self.configs.features == "MS" else 0
        out = {"pred": pred[:, -y.shape[1]:, f_dim:], "true": y[:, :, f_dim:], "mask": y_mask[:, :, f_dim:]}
        self.latest_diag = {
            "field_density": float(stat[..., 0].mean().detach().cpu()),
            "field_recency": float(stat[..., 6].mean().detach().cpu()),
        }
        return out
