import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.ExpConfigs import ExpConfigs


def _time(length: int, batch: int, device, dtype) -> Tensor:
    denom = max(length - 1, 1)
    t = torch.arange(length, device=device, dtype=dtype) / denom
    return t.view(1, length, 1).expand(batch, -1, -1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FourierTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, max(dim - 1, 1))
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        out = torch.cat([self.linear(t), torch.sin(self.periodic(t))], dim=-1)
        return out[..., : self.dim]


class Model(nn.Module):
    """Arrival Mechanism Patch Graph Network.

    AMPG-Net is a native IMTS forecasting backbone. It first maps irregular
    observations into mechanism-aware patch tokens: a local operator preserves
    the latest arrival state, and an integral operator summarizes the interval
    evidence inside the same patch. A closed gate forms the patch token before
    temporal and variable graph mixing, so the mechanism is part of the backbone
    rather than a residual correction after another model has forecasted.
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
        self.aux_weight = 0.01 if "no_aux" not in self.mode else 0.0

        self.var_emb = nn.Parameter(torch.randn(self.n_vars, self.d_model) * 0.02)
        self.time_enc = FourierTime(max(4, int(getattr(configs, "tpatchgnn_te_dim", 10))))
        te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
        stat_dim = 12
        token_in = stat_dim + te_dim + self.d_model
        self.local_op = MLP(token_in, self.hidden, self.d_model, self.dropout)
        self.integral_op = MLP(token_in, self.hidden, self.d_model, self.dropout)
        self.mechanism_gate = MLP(token_in + self.d_model * 2, self.hidden, self.d_model, self.dropout)
        self.patch_norm = nn.LayerNorm(self.d_model)

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
        self.temporal_pos = nn.Parameter(torch.randn(1, 512, self.d_model) * 0.01)

        graph_dim = max(8, int(getattr(configs, "node_dim", 10)))
        self.graph_q = nn.Linear(self.d_model, graph_dim)
        self.graph_k = nn.Linear(self.d_model, graph_dim)
        self.graph_msg = nn.Linear(self.d_model, self.d_model)
        self.graph_gate = MLP(self.d_model * 2 + stat_dim, self.hidden, self.d_model, self.dropout)
        self.graph_norm = nn.LayerNorm(self.d_model)

        self.pool_score = nn.Linear(self.d_model + stat_dim, 1)
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
        self.future_mean = nn.Linear(self.d_model, 1)
        # Initialized after the original AMPG modules so full mode keeps the old random initialization order.
        self.alias_gate = MLP(stat_dim + self.d_model * 3, self.hidden, self.d_model, self.dropout)
        self.regime_gate = MLP(stat_dim + self.d_model * 5, self.hidden, self.d_model, self.dropout)
        self.single_op = MLP(token_in, self.hidden, self.d_model, self.dropout)
        self.latest_diag = {}

    def _get_time(self, mark: Tensor | None, ref: Tensor) -> Tensor:
        if mark is None:
            return _time(ref.size(1), ref.size(0), ref.device, ref.dtype)
        if mark.dim() == 2:
            mark = mark.unsqueeze(-1)
        return torch.nan_to_num(mark[:, :, [0]].to(dtype=ref.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _pad(self, x: Tensor, value: float = 0.0) -> Tensor:
        rem = x.size(1) % self.patch_len
        if rem == 0:
            return x
        pad = self.patch_len - rem
        return F.pad(x, (0, 0, 0, pad), value=value)

    def _patch_stats(self, x: Tensor, mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, n_vars = x.shape
        x = self._pad(x, 0.0)
        mask = self._pad(mask, 0.0)
        t = self._pad(t, 1.0)
        n_patch = x.size(1) // self.patch_len
        x = x.view(batch, n_patch, self.patch_len, n_vars).permute(0, 3, 1, 2)
        mask = mask.view(batch, n_patch, self.patch_len, n_vars).permute(0, 3, 1, 2)
        t = t.view(batch, n_patch, self.patch_len, 1).permute(0, 3, 1, 2).expand(-1, n_vars, -1, -1)

        denom = mask.sum(dim=-1).clamp_min(1.0)
        density = mask.mean(dim=-1)
        mean = (x * mask).sum(dim=-1) / denom
        second = (x.square() * mask).sum(dim=-1) / denom
        var = (second - mean.square()).clamp_min(0.0)
        first_idx = mask.argmax(dim=-1)
        rev_idx = torch.flip(mask, dims=[-1]).argmax(dim=-1)
        last_idx = self.patch_len - 1 - rev_idx
        has_obs = (mask.sum(dim=-1) > 0).to(x.dtype)
        gather_first = first_idx.unsqueeze(-1)
        gather_last = last_idx.unsqueeze(-1)
        first_x = torch.gather(x, -1, gather_first).squeeze(-1) * has_obs
        last_x = torch.gather(x, -1, gather_last).squeeze(-1) * has_obs
        first_t = torch.gather(t, -1, gather_first).squeeze(-1) * has_obs
        last_t = torch.gather(t, -1, gather_last).squeeze(-1) * has_obs
        patch_start = t[..., 0]
        patch_end = t[..., -1]
        span = (last_t - first_t).clamp_min(0.0)
        recency = (patch_end - last_t).clamp_min(0.0)
        slope = (last_x - first_x) / span.clamp_min(1e-3)
        center_t = 0.5 * (patch_start + patch_end)
        stat = torch.stack(
            [density, has_obs, mean, var, first_x, last_x, slope, first_t, last_t, span, recency, center_t],
            dim=-1,
        )
        return stat, center_t.unsqueeze(-1), has_obs

    def _make_tokens(self, stat: Tensor, center_t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, n_vars, n_patch, _ = stat.shape
        var = self.var_emb.to(stat.device, stat.dtype).view(1, n_vars, 1, self.d_model).expand(batch, -1, n_patch, -1)
        te = self.time_enc(center_t)
        base = torch.cat([stat, te, var], dim=-1)
        local = self.local_op(base)
        integral = self.integral_op(base)
        diff = (local - integral).abs()
        closed_gate = torch.sigmoid(self.mechanism_gate(torch.cat([base, local, integral], dim=-1)))
        alias_gate = torch.sigmoid(self.alias_gate(torch.cat([stat, local, integral, diff], dim=-1)))
        regime_gate = torch.zeros_like(local)
        if "single_token" in self.mode or "one_token" in self.mode:
            token = self.single_op(base)
            gate = torch.zeros_like(local)
        elif "local_only" in self.mode:
            token = local
            gate = torch.zeros_like(local)
        elif "integral_only" in self.mode:
            token = integral
            gate = torch.ones_like(integral)
        elif "no_mechanism_gate" in self.mode or "mean_gate" in self.mode:
            gate = torch.full_like(local, 0.5)
            token = 0.5 * local + 0.5 * integral
        elif "alias_aware" in self.mode or "alias_gate" in self.mode:
            gate = alias_gate
            token = local + gate * (integral - local)
        elif "dual_gate" in self.mode or "hybrid_gate" in self.mode or "fused_gate" in self.mode:
            regime_gate = torch.sigmoid(self.regime_gate(torch.cat([stat, local, integral, diff, closed_gate, alias_gate], dim=-1)))
            gate = (1.0 - regime_gate) * closed_gate + regime_gate * alias_gate
            token = local + gate * (integral - local)
        else:
            gate = closed_gate
            token = (1.0 - gate) * local + gate * integral
        return self.patch_norm(token + var), local, gate, alias_gate, regime_gate

    def _temporal_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        batch, n_vars, n_patch, dim = token.shape
        pos = self.temporal_pos[:, :n_patch, :].to(token.device, token.dtype)
        h = token.reshape(batch * n_vars, n_patch, dim) + pos
        h = self.temporal(h).view(batch, n_vars, n_patch, dim)
        if "no_graph" in self.mode:
            return h
        q = self.graph_q(h).permute(0, 2, 1, 3)
        k = self.graph_k(h).permute(0, 2, 1, 3)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1)), dim=-1)
        msg = torch.einsum("bpij,bjpd->bipd", attn, h)
        msg = self.graph_msg(msg)
        gate = torch.sigmoid(self.graph_gate(torch.cat([h, msg, stat], dim=-1)))
        if alias_gate is not None and ("alias_aware" in self.mode or "alias_graph" in self.mode or "dual_gate" in self.mode or "hybrid_gate" in self.mode or "fused_gate" in self.mode):
            alias_strength = alias_gate.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
            gate = gate * (0.25 + 0.75 * alias_strength)
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
        batch, seq_len, n_vars = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x_mask = torch.nan_to_num(x_mask.to(dtype=x.dtype), nan=0.0).clamp(0.0, 1.0)
        if y is None:
            y = torch.zeros(batch, self.pred_len, n_vars, dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y)
        t = self._get_time(x_mark, x)
        y_t = self._get_time(y_mark, y) if y_mark is not None else _time(y.size(1), batch, x.device, x.dtype)

        stat, center_t, has_obs = self._patch_stats(x, x_mask, t)
        stat_diag = stat
        state_cols = [0, 1, 7, 8, 9, 10]
        stat_pregraph = stat
        stat_decoder = stat
        if "random_state" in self.mode or "shuffle_state" in self.mode:
            # Preserve the marginal mechanism-state distribution but break its sample association.
            stat_pregraph = stat.clone()
            if stat_pregraph.size(0) > 1:
                perm = torch.randperm(stat_pregraph.size(0), device=stat_pregraph.device)
                stat_pregraph[..., state_cols] = stat_pregraph[perm][..., state_cols]
            stat_decoder = stat_pregraph
        if "no_state" in self.mode or "value_only_state" in self.mode or "post_graph_state" in self.mode:
            # Keep value-level patch summaries but remove deployment-visible arrival/mechanism state before token/graph formation.
            stat_pregraph = stat_pregraph.clone()
            stat_pregraph[..., state_cols] = 0.0
            if "post_graph_state" not in self.mode:
                stat_decoder = stat_pregraph
        token, local, mech_gate, alias_gate, regime_gate = self._make_tokens(stat_pregraph, center_t)
        graph_signal = mech_gate if ("dual_gate" in self.mode or "hybrid_gate" in self.mode or "fused_gate" in self.mode) else alias_gate
        h = self._temporal_graph(token, stat_pregraph, graph_signal)

        score = self.pool_score(torch.cat([h, stat_pregraph], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        enc = (weights * h).sum(dim=2)

        # Carry a compact deployment-valid mechanism state into the decoder.
        last_stat = stat_decoder[:, :, -1, :]
        global_state = enc.mean(dim=1, keepdim=True).expand(-1, n_vars, -1)
        enc_exp = enc.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        glob_exp = global_state.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        te_f = self.future_te(y_t).unsqueeze(1).expand(-1, n_vars, -1, -1)
        tau = (y_t[:, :, 0] - t[:, -1, 0].view(batch, 1)).clamp_min(0.0)
        dec_mech = torch.stack(
            [
                tau.unsqueeze(1).expand(-1, n_vars, -1),
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
        dec_in = torch.cat([enc_exp, glob_exp, te_f, dec_mech], dim=-1)
        pred = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)

        f_dim = -1 if self.configs.features == "MS" else 0
        out = {"pred": pred[:, -y.shape[1]:, f_dim:], "true": y[:, :, f_dim:], "mask": y_mask[:, :, f_dim:]}
        if self.training and self.aux_weight > 0.0:
            ym = y_mask.to(dtype=y.dtype)
            denom = ym.sum(dim=1).clamp_min(1.0)
            future_mean = (y * ym).sum(dim=1) / denom
            out["aux_loss"] = self.aux_weight * F.l1_loss(self.future_mean(enc).squeeze(-1), future_mean)
        self.latest_diag = {
            "mechanism_gate_mean": float(mech_gate.mean().detach().cpu()),
            "patch_density": float(stat_diag[..., 0].mean().detach().cpu()),
            "alias_gate_mean": float(alias_gate.mean().detach().cpu()),
            "regime_gate_mean": float(regime_gate.mean().detach().cpu()),
        }
        return out






