import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from models.CTPGNet import Model as CTPGModel
from utils.ExpConfigs import ExpConfigs


class Model(CTPGModel):
    """Regime-Adaptive Continuous/Mechanism Patch Graph Network.

    RACPG-Net is a coordinate-adaptive IMTS backbone. It constructs two
    deployment-valid patch coordinates before any prediction is made:

    1. fixed mechanism-state patches from local/integral summary statistics;
    2. continuous-time kernel patches from irregular observations.

    A learned coordinate gate fuses the two patch coordinates at token level,
    then a single temporal/variable graph backbone forecasts the future. This
    is not a residual wrapper or a prediction ensemble: the choice happens
    before temporal and cross-variable dependency learning.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        route_in = 36 + 3 * self.d_model
        self.coord_scalar = MLP(route_in, self.hidden, 1, self.dropout)
        self.coord_vector = MLP(route_in, self.hidden, self.d_model, self.dropout)
        self.coord_norm = nn.LayerNorm(self.d_model)
        if "fixed_prior" in self.mode:
            last = self.coord_scalar.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.constant_(last.bias, -1.0)
        elif "ct_prior" in self.mode:
            last = self.coord_scalar.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.constant_(last.bias, 1.0)

    def _fixed_coordinate(self, x: Tensor, x_mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        stat, center_t, has_obs = AMPGModel._patch_stats(self, x, x_mask, t)
        token, _, _, alias_gate, _ = AMPGModel._make_tokens(self, stat, center_t)
        return token, stat, has_obs, alias_gate

    def _ct_coordinate(self, x: Tensor, x_mask: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        stat, center_t, has_obs = CTPGModel._patch_stats(self, x, x_mask, t)
        token, _, _, alias_gate, _ = CTPGModel._make_tokens(self, stat, center_t)
        ct_stat = self._ct_fused_stat if self._ct_fused_stat is not None else stat
        return token, ct_stat, has_obs, alias_gate

    def _route(
        self,
        fixed_token: Tensor,
        fixed_stat: Tensor,
        ct_token: Tensor,
        ct_stat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        route = torch.cat(
            [
                fixed_stat,
                ct_stat,
                (fixed_stat - ct_stat).abs(),
                fixed_token,
                ct_token,
                (fixed_token - ct_token).abs(),
            ],
            dim=-1,
        )
        scalar = torch.sigmoid(self.coord_scalar(route))
        if "mean_route" in self.mode:
            scalar = torch.full_like(scalar, 0.5)
            vector = scalar.expand_as(fixed_token)
        elif "scalar_route" in self.mode or "fixed_prior" in self.mode or "ct_prior" in self.mode:
            vector = scalar.expand_as(fixed_token)
        else:
            vector = torch.sigmoid(self.coord_vector(route))
            vector = 0.5 * vector + 0.5 * scalar

        if "fixed_only" in self.mode:
            vector = torch.zeros_like(fixed_token)
            scalar = torch.zeros_like(scalar)
        elif "ct_only" in self.mode:
            vector = torch.ones_like(fixed_token)
            scalar = torch.ones_like(scalar)

        token = (1.0 - vector) * fixed_token + vector * ct_token
        stat = (1.0 - scalar) * fixed_stat + scalar * ct_stat
        return self.coord_norm(token), stat, scalar

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
        y_t = self._get_time(y_mark, y) if y_mark is not None else _time(y.size(1), batch, x.device, x.dtype)

        fixed_token, fixed_stat, fixed_obs, fixed_alias = self._fixed_coordinate(x, x_mask, t)
        ct_token, ct_stat, ct_obs, ct_alias = self._ct_coordinate(x, x_mask, t)
        token, stat, coord_gate = self._route(fixed_token, fixed_stat, ct_token, ct_stat)
        has_obs = torch.maximum(fixed_obs, ct_obs)
        alias_gate = 0.5 * (fixed_alias + ct_alias)

        h = AMPGModel._temporal_graph(self, token, stat, alias_gate)
        if "no_stconv" not in self.mode:
            msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            gate = torch.sigmoid(self.ct_st_gate(torch.cat([stat, h], dim=-1)))
            h = self.ct_st_norm(h + gate * msg)

        score = self.pool_score(torch.cat([h, stat], dim=-1)).squeeze(-1)
        score = score.masked_fill(has_obs <= 0, -1e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        enc = (weights * h).sum(dim=2)

        last_stat = stat[:, :, -1, :]
        global_state = enc.mean(dim=1, keepdim=True).expand(-1, self.n_vars, -1)
        enc_exp = enc.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        glob_exp = global_state.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        te_f = self.future_te(y_t).unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        tau = (y_t[:, :, 0] - t[:, -1, 0].view(batch, 1)).clamp_min(0.0)
        dec_mech = torch.stack(
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
        dec_in = torch.cat([enc_exp, glob_exp, te_f, dec_mech], dim=-1)
        pred = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1)

        f_dim = -1 if self.configs.features == "MS" else 0
        out = {"pred": pred[:, -y.shape[1]:, f_dim:], "true": y[:, :, f_dim:], "mask": y_mask[:, :, f_dim:]}
        if self.training and "no_prob" not in self.mode:
            sigma = F.softplus(self.ct_sigma_decoder(dec_in).squeeze(-1).permute(0, 2, 1)) + 1e-3
            sigma = sigma[:, -y.shape[1]:, f_dim:]
            mask = y_mask[:, :, f_dim:].to(dtype=pred.dtype)
            residual = (out["pred"] - y[:, :, f_dim:]) * mask
            nll = 0.5 * (residual.square() / sigma.square() + 2.0 * torch.log(sigma)) * mask
            out["aux_loss"] = 0.0005 * nll.sum() / mask.sum().clamp_min(1.0)
        self.latest_diag = {
            "coord_gate_mean": float(coord_gate.mean().detach().cpu()),
            "patch_density": float(stat[..., 0].mean().detach().cpu()),
            "alias_gate_mean": float(alias_gate.mean().detach().cpu()),
            "ct_width_mean": float(self.ct_log_width.exp().mean().detach().cpu()),
        }
        return out
