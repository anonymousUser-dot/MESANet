import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from utils.ExpConfigs import ExpConfigs


class Model(AMPGModel):
    """Amplitude-Preserving Event-Scan Network.

    APESNet is a native forecasting backbone for irregular wearable time
    series. It keeps the observed value level as an explicit causal state and
    lets neural operators learn only how that state should be propagated,
    mixed across variables, and displaced into the forecast horizon. This tests
    the first-principles hypothesis left by CEFONet: an IMTS backbone may smooth
    mechanism information, but it must not replace the amplitude coordinate.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        te_dim = max(4, int(getattr(configs, "tpatchgnn_te_dim", 10)))
        self.apes_in = MLP(12 + te_dim + self.d_model, self.hidden, self.d_model, self.dropout)
        self.apes_gate = MLP(12 + self.d_model * 2, self.hidden, self.d_model, self.dropout)
        self.apes_drift = MLP(12 + self.d_model + 1, self.hidden, 1, self.dropout)
        self.apes_norm = nn.LayerNorm(self.d_model)
        self.apes_graph_gate = MLP(12 + self.d_model * 2, self.hidden, self.d_model, self.dropout)
        self.apes_delta_scale = MLP(8 + self.d_model, self.hidden, 1, self.dropout)

    def _scan(self, stat: Tensor, center_t: Tensor, has_obs: Tensor) -> tuple[Tensor, Tensor]:
        batch, n_vars, n_patch, _ = stat.shape
        var = self.var_emb.to(stat.device, stat.dtype).view(1, n_vars, 1, self.d_model).expand(batch, -1, n_patch, -1)
        te = self.time_enc(center_t)
        drive = self.apes_in(torch.cat([stat, te, var], dim=-1))

        h = torch.zeros(batch, n_vars, self.d_model, device=stat.device, dtype=stat.dtype)
        level = stat[:, :, 0, 5]
        prev_t = stat[:, :, 0, 11]
        states = []
        levels = []
        for p in range(n_patch):
            stat_p = stat[:, :, p, :]
            drive_p = drive[:, :, p, :]
            obs = has_obs[:, :, p].unsqueeze(-1)
            gate = torch.sigmoid(self.apes_gate(torch.cat([stat_p, h, drive_p], dim=-1)))
            if "open_scan" not in self.mode:
                gate = 0.25 * gate + 0.75 * gate * obs
            if "no_scan_gate" in self.mode:
                gate = torch.ones_like(gate) * obs
            h = self.apes_norm(h + gate * (drive_p - h))

            dt = (stat_p[..., 11] - prev_t).clamp_min(0.0).unsqueeze(-1)
            drift = torch.tanh(self.apes_drift(torch.cat([stat_p, h, dt], dim=-1))).squeeze(-1)
            if "no_drift" in self.mode:
                drift = torch.zeros_like(drift)
            observed_level = stat_p[..., 5]
            level = obs.squeeze(-1) * observed_level + (1.0 - obs.squeeze(-1)) * (level + 0.05 * drift)
            prev_t = stat_p[..., 11]
            states.append(h)
            levels.append(level)
        return torch.stack(states, dim=2), torch.stack(levels, dim=2)

    def _graph_mix(self, h: Tensor, stat: Tensor) -> Tensor:
        if "no_graph" in self.mode:
            return h
        q = self.graph_q(h).permute(0, 2, 1, 3)
        k = self.graph_k(h).permute(0, 2, 1, 3)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5), dim=-1)
        msg = torch.einsum("bpij,bjpd->bipd", attn, h)
        msg = self.graph_msg(msg)
        gate = torch.sigmoid(self.apes_graph_gate(torch.cat([stat, h, msg], dim=-1)))
        if "closed_graph" in self.mode:
            density = stat[..., 0:1].clamp(0.0, 1.0)
            gate = gate * (0.25 + 0.75 * density)
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
        batch, _, n_vars = x.shape
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
        h, levels = self._scan(stat, center_t, has_obs)
        h = self._graph_mix(h, stat)

        score = self.pool_score(torch.cat([h, stat], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        enc = (weights * h).sum(dim=2)

        last_stat = stat[:, :, -1, :]
        last_level = levels[:, :, -1]
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
        raw = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)
        if "direct_decoder" in self.mode:
            pred = raw
        else:
            scale_in = torch.cat([dec_mech, enc_exp], dim=-1)
            amp = F.softplus(self.apes_delta_scale(scale_in).squeeze(-1).permute(0, 2, 1)) + 1.0e-3
            base_amp = last_stat[..., 3].sqrt().unsqueeze(1) + last_stat[..., 6].abs().unsqueeze(1) * tau.unsqueeze(-1)
            amp = amp * (0.05 + base_amp.clamp(0.0, 2.0))
            if "free_delta" in self.mode:
                delta = raw
            else:
                delta = torch.tanh(raw) * amp
            pred = last_level.unsqueeze(1) + delta

        f_dim = -1 if self.configs.features == "MS" else 0
        out = {"pred": pred[:, -y.shape[1]:, f_dim:], "true": y[:, :, f_dim:], "mask": y_mask[:, :, f_dim:]}
        if self.training and self.aux_weight > 0.0 and "no_aux" not in self.mode:
            ym = y_mask.to(dtype=y.dtype)
            denom = ym.sum(dim=1).clamp_min(1.0)
            future_mean = (y * ym).sum(dim=1) / denom
            out["aux_loss"] = self.aux_weight * F.l1_loss(self.future_mean(enc).squeeze(-1), future_mean)
        self.latest_diag = {
            "patch_density": float(stat[..., 0].mean().detach().cpu()),
            "level_abs": float(last_level.abs().mean().detach().cpu()),
        }
        return out
