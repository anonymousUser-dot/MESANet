import torch
import torch.nn as nn
from torch import Tensor

from models.AMPGNet import MLP, Model as AMPGModel
from models.MCGONet import Model as MCGOModel
from utils.ExpConfigs import ExpConfigs


class Model(MCGOModel):
    """Dependency-Admitted Graph Operator Network.

    DAGONet is a native IMTS forecasting backbone with two closed admissions:
    temporal-scale admission from CAGNet and dependency-operator admission here.
    The second admission keeps the CAGNet content graph as the default operator
    and admits a mechanism-conditioned graph only where deployment-visible
    mechanism state indicates that content-only variable exchange is aliased.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)
        route_dim = 36 + 3 * self.d_model
        self.dep_admit_scalar = MLP(route_dim, self.hidden, 1, self.dropout)
        self.dep_admit_vector = MLP(route_dim, self.hidden, self.d_model, self.dropout)
        self.dep_admit_norm = nn.LayerNorm(self.d_model)
        scalar_last = self.dep_admit_scalar.net[-1]
        vector_last = self.dep_admit_vector.net[-1]
        if isinstance(scalar_last, nn.Linear):
            bias = -2.0
            if "open_dep" in self.mode:
                bias = -0.5
            elif "very_closed_dep" in self.mode:
                bias = -3.0
            nn.init.constant_(scalar_last.bias, bias)
        if isinstance(vector_last, nn.Linear):
            bias = -2.0
            if "open_dep" in self.mode:
                bias = -0.5
            elif "very_closed_dep" in self.mode:
                bias = -3.0
            nn.init.constant_(vector_last.bias, bias)

    def _content_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        h = AMPGModel._temporal_graph(self, token, stat, alias_gate)
        if "no_stconv" not in self.mode:
            msg = self.ct_st_conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            gate = torch.sigmoid(self.ct_st_gate(torch.cat([stat, h], dim=-1)))
            h = self.ct_st_norm(h + gate * msg)
        return h

    @staticmethod
    def _mechanism_context(stat: Tensor) -> tuple[Tensor, Tensor]:
        center = stat.mean(dim=1, keepdim=True).expand_as(stat)
        deviation = (stat - center).abs()
        return center, deviation

    def _mechanism_graph(self, token: Tensor, stat: Tensor, alias_gate: Tensor | None = None) -> Tensor:
        if "no_dep_admit" in self.mode:
            return super()._mechanism_graph(token, stat, alias_gate)

        content_h = self._content_graph(token, stat, alias_gate)
        mech_h = super()._mechanism_graph(token, stat, alias_gate)
        center, deviation = self._mechanism_context(stat)
        route = torch.cat([stat, center, deviation, content_h, mech_h, (content_h - mech_h).abs()], dim=-1)
        scalar = torch.sigmoid(self.dep_admit_scalar(route))
        if "scalar_dep" in self.mode or "very_closed_dep" in self.mode or "open_dep" in self.mode:
            vector = scalar.expand_as(content_h)
        else:
            vector = torch.sigmoid(self.dep_admit_vector(route))
            vector = 0.5 * vector + 0.5 * scalar
        if alias_gate is not None and "alias_dep" in self.mode:
            strength = alias_gate.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
            vector = vector * (0.25 + 0.75 * strength)
        if "content_only" in self.mode:
            vector = torch.zeros_like(vector)
        elif "mechanism_only" in self.mode:
            vector = torch.ones_like(vector)
        h = content_h + vector * (mech_h - content_h)
        self.latest_dep_admit = vector.detach()
        if "dep_norm" in self.mode:
            h = self.dep_admit_norm(h)
        return h
