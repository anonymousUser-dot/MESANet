import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel, _time
from models.CTPGNet import Model as CTPGModel
from models.RACPGNet import Model as RACPGModel
from utils.ExpConfigs import ExpConfigs


class Model(RACPGModel):
    """State-Resolution Patch Graph Network.

    SRPG-Net upgrades RACPG/MRPS from a structural state split to a trained
    state-resolution backbone. It keeps the end-to-end forecasting path, but
    adds training-only supervision that makes mechanism and continuous-time
    coordinates predict future summary targets before graph transfer. The
    coordinate gate is then trained to route toward the coordinate with lower
    summary risk. At deployment the model uses only input-visible patch states.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        self.state_aux_weight = 0.01
        self.route_aux_weight = 0.005
        if "strong_state_aux" in self.mode:
            self.state_aux_weight = 0.02
            self.route_aux_weight = 0.01
        if "weak_state_aux" in self.mode:
            self.state_aux_weight = 0.005
            self.route_aux_weight = 0.0025
        if "no_state_aux" in self.mode:
            self.state_aux_weight = 0.0
        if "no_route_aux" in self.mode:
            self.route_aux_weight = 0.0

        self.fixed_summary_head = MLP(self.d_model, self.hidden, 1, self.dropout)
        self.ct_summary_head = MLP(self.d_model, self.hidden, 1, self.dropout)
        self.fused_summary_head = MLP(self.d_model, self.hidden, 1, self.dropout)

    @staticmethod
    def _masked_patch_pool(token: Tensor, has_obs: Tensor) -> Tensor:
        weights = has_obs.to(dtype=token.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (weights.unsqueeze(-1) * token).sum(dim=2)

    def _resolution_aux(
        self,
        fixed_token: Tensor,
        fixed_obs: Tensor,
        ct_token: Tensor,
        ct_obs: Tensor,
        fused_token: Tensor,
        fused_obs: Tensor,
        coord_gate: Tensor,
        y: Tensor,
        y_mask: Tensor,
        f_dim,
    ) -> tuple[Tensor, dict[str, float]]:
        y_true = y[:, :, f_dim:]
        mask = y_mask[:, :, f_dim:].to(dtype=y_true.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        future_mean = (y_true * mask).sum(dim=1) / denom

        fixed_pool = self._masked_patch_pool(fixed_token, fixed_obs)
        ct_pool = self._masked_patch_pool(ct_token, ct_obs)
        fused_pool = self._masked_patch_pool(fused_token, fused_obs)
        fixed_mu = self.fixed_summary_head(fixed_pool).squeeze(-1)
        ct_mu = self.ct_summary_head(ct_pool).squeeze(-1)
        fused_mu = self.fused_summary_head(fused_pool).squeeze(-1)

        state_aux = (
            F.smooth_l1_loss(fixed_mu, future_mean)
            + F.smooth_l1_loss(ct_mu, future_mean)
            + F.smooth_l1_loss(fused_mu, future_mean)
        )

        fixed_err = (fixed_mu.detach() - future_mean).abs()
        ct_err = (ct_mu.detach() - future_mean).abs()
        temp = 0.05 if "sharp_route_target" in self.mode else 0.10
        route_target = torch.sigmoid((fixed_err - ct_err) / temp)
        gate = self._masked_patch_pool(coord_gate.expand_as(fused_token), fused_obs).mean(dim=-1).clamp(1e-4, 1 - 1e-4)
        route_aux = F.binary_cross_entropy_with_logits(torch.logit(gate.float()), route_target.float())

        aux = self.state_aux_weight * state_aux + self.route_aux_weight * route_aux
        diag = {
            "sr_state_aux": float(state_aux.detach().cpu()),
            "sr_route_aux": float(route_aux.detach().cpu()),
            "sr_route_target_mean": float(route_target.mean().detach().cpu()),
        }
        return aux, diag

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
        diag_extra = {}
        if self.training:
            aux, diag_extra = self._resolution_aux(
                fixed_token=fixed_token,
                fixed_obs=fixed_obs,
                ct_token=ct_token,
                ct_obs=ct_obs,
                fused_token=token,
                fused_obs=has_obs,
                coord_gate=coord_gate,
                y=y,
                y_mask=y_mask,
                f_dim=f_dim,
            )
            if "no_prob" not in self.mode:
                sigma = F.softplus(self.ct_sigma_decoder(dec_in).squeeze(-1).permute(0, 2, 1)) + 1e-3
                sigma = sigma[:, -y.shape[1]:, f_dim:]
                mask = y_mask[:, :, f_dim:].to(dtype=pred.dtype)
                residual = (out["pred"] - y[:, :, f_dim:]) * mask
                nll = 0.5 * (residual.square() / sigma.square() + 2.0 * torch.log(sigma)) * mask
                aux = aux + 0.0005 * nll.sum() / mask.sum().clamp_min(1.0)
            out["aux_loss"] = aux
        self.latest_diag = {
            "coord_gate_mean": float(coord_gate.mean().detach().cpu()),
            "patch_density": float(stat[..., 0].mean().detach().cpu()),
            "alias_gate_mean": float(alias_gate.mean().detach().cpu()),
            "ct_width_mean": float(self.ct_log_width.exp().mean().detach().cpu()),
            **diag_extra,
        }
        return out
