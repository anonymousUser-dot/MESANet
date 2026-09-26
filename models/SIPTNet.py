"""Support-Identifiable Patch Transformer for wearable IMTS forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from layers.Formers.Embed import PositionalEmbedding
from layers.Formers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Formers.Transformer_EncDec import Encoder, EncoderLayer
from utils.ExpConfigs import ExpConfigs


class Transpose(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, in_features: int, pred_len: int, dropout: float):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, pred_len)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.dropout(self.flatten(x)))


class SupportPatchEmbedding(nn.Module):
    """Embed value paths jointly with deployment-visible observation support."""

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.support_rank = min(4, patch_len)
        position = torch.arange(patch_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.arange(self.support_rank, dtype=torch.float32).unsqueeze(0)
        basis = torch.cos(math.pi * (position + 0.5) * frequency / float(patch_len))
        basis[:, 0] *= math.sqrt(1.0 / float(patch_len))
        if self.support_rank > 1:
            basis[:, 1:] *= math.sqrt(2.0 / float(patch_len))
        self.register_buffer("support_basis", basis, persistent=True)
        self.value_content = nn.Linear(patch_len, d_model, bias=False)
        self.support_content = nn.Linear(3 * self.support_rank, d_model, bias=False)
        nn.init.zeros_(self.support_content.weight)
        self.support_modulation = nn.Linear(4, 2 * d_model)
        nn.init.zeros_(self.support_modulation.weight)
        nn.init.zeros_(self.support_modulation.bias)
        self.position = PositionalEmbedding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: Tensor,
        mask: Tensor,
        velocity: Tensor,
        age: Tensor,
        mode: str,
    ) -> tuple[Tensor, int, Tensor]:
        # Inputs are [B,V,L]. Padding carries no fabricated observations.
        value = F.pad(value, (0, self.stride), value=0.0)
        mask = F.pad(mask, (0, self.stride), value=0.0)
        velocity = F.pad(velocity, (0, self.stride), value=0.0)
        age = F.pad(age, (0, self.stride), value=1.0)
        value = value.unfold(-1, self.patch_len, self.stride)
        mask = mask.unfold(-1, self.patch_len, self.stride)
        velocity = velocity.unfold(-1, self.patch_len, self.stride)
        age = age.unfold(-1, self.patch_len, self.stride)

        if "value_only" in mode:
            mask_input = torch.zeros_like(mask)
            velocity_input = torch.zeros_like(velocity)
            age_input = torch.zeros_like(age)
        else:
            mask_input = mask
            velocity_input = torch.zeros_like(velocity) if "no_velocity" in mode else velocity
            age_input = torch.zeros_like(age) if "no_age" in mode else age

        token = self.value_content(value)
        if "value_only" not in mode:
            basis = self.support_basis.to(device=value.device, dtype=value.dtype)
            support_path = torch.cat(
                [
                    torch.einsum("bvnp,pr->bvnr", mask_input, basis),
                    torch.einsum("bvnp,pr->bvnr", velocity_input, basis),
                    torch.einsum("bvnp,pr->bvnr", age_input, basis),
                ],
                dim=-1,
            )
            token = token + 0.25 * self.support_content(support_path)

        position = torch.linspace(0.0, 1.0, self.patch_len, device=mask.device, dtype=mask.dtype)
        observed = mask > 0
        first = torch.where(observed, position, torch.ones_like(position)).amin(dim=-1)
        last = torch.where(observed, position, torch.zeros_like(position)).amax(dim=-1)
        any_observed = observed.any(dim=-1)
        density = mask.mean(dim=-1)
        span = torch.where(any_observed, last - first, torch.zeros_like(last))
        recency = torch.where(any_observed, 1.0 - last, torch.ones_like(last))
        mean_age = (age * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        support = torch.stack([density, span, recency, mean_age], dim=-1)

        if "no_support_mod" not in mode and "value_only" not in mode:
            gamma, beta = self.support_modulation(support).chunk(2, dim=-1)
            token = (1.0 + 0.25 * torch.tanh(gamma)) * token + 0.25 * beta
        batch, n_vars, n_patch, dim = token.shape
        token = token.reshape(batch * n_vars, n_patch, dim)
        token = self.norm(token + self.position(token))
        return self.dropout(token), n_vars, support


class Model(nn.Module):
    """End-to-end mask/time-aware channel-independent forecasting backbone."""

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.mode = str(getattr(configs, "ablation_name", "") or "").lower()
        self.seq_len = configs.seq_len_max_irr or configs.seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.patch_len = max(2, int(configs.patch_len_max_irr or configs.patch_len))
        self.stride = max(1, int(configs.patch_stride))
        self.n_vars = int(configs.enc_in)
        self.d_model = int(configs.d_model)
        self.patch_embedding = SupportPatchEmbedding(
            self.patch_len, self.stride, self.d_model, float(configs.dropout)
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        self.d_model,
                        configs.n_heads,
                    ),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2)
            ),
        )
        self.n_patch = (self.seq_len + self.stride - self.patch_len) // self.stride + 1
        self.head = FlattenHead(
            self.n_vars,
            self.d_model * self.n_patch,
            self.pred_len,
            float(configs.dropout),
        )

    @staticmethod
    def _time(mark: Tensor | None, ref: Tensor) -> Tensor:
        if mark is None:
            time = torch.linspace(0.0, 1.0, ref.size(1), device=ref.device, dtype=ref.dtype)
            return time.view(1, -1).expand(ref.size(0), -1)
        if mark.dim() == 2:
            return mark.to(dtype=ref.dtype).clamp(0.0, 1.0)
        return mark[:, :, 0].to(dtype=ref.dtype).clamp(0.0, 1.0)

    @staticmethod
    def _masked_anchor(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (x * mask).sum(dim=1) / count
        centered = (x - mean.unsqueeze(1)) * mask
        scale = torch.sqrt(centered.square().sum(dim=1) / count + 1e-5).clamp_min(0.05)
        return centered / scale.unsqueeze(1), mean, scale

    @staticmethod
    def _observation_age(time: Tensor, mask: Tensor) -> Tensor:
        expanded_time = time.unsqueeze(-1).expand_as(mask)
        sentinel = torch.full_like(expanded_time, -2.0)
        observed_time = torch.where(mask > 0, expanded_time, sentinel)
        last_time = torch.cummax(observed_time, dim=1).values
        history_span = (time[:, -1] - time[:, 0]).clamp_min(1e-5).view(-1, 1, 1)
        age = (expanded_time - last_time) / history_span
        return torch.where(last_time > -1.0, age, torch.ones_like(age)).clamp(0.0, 1.0)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor = None,
        x_mask: Tensor = None,
        y: Tensor = None,
        y_mask: Tensor = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        del kwargs
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x_mask = torch.nan_to_num(x_mask.to(dtype=x.dtype), nan=0.0).clamp(0.0, 1.0)
        if y is None:
            y = torch.zeros(x.size(0), self.pred_len, self.n_vars, device=x.device, dtype=x.dtype)
        if y_mask is None:
            y_mask = torch.ones_like(y)

        normalized, mean, scale = self._masked_anchor(x, x_mask)
        time = self._time(x_mark, x)
        age = self._observation_age(time, x_mask)
        pair = x_mask[:, 1:, :] * x_mask[:, :-1, :]
        velocity = F.pad((normalized[:, 1:, :] - normalized[:, :-1, :]) * pair, (0, 0, 1, 0))

        token, n_vars, support = self.patch_embedding(
            normalized.permute(0, 2, 1),
            x_mask.permute(0, 2, 1),
            velocity.permute(0, 2, 1),
            age.permute(0, 2, 1),
            self.mode,
        )
        encoded, _ = self.encoder(token)
        encoded = encoded.reshape(x.size(0), n_vars, self.n_patch, self.d_model)
        encoded = encoded.permute(0, 1, 3, 2)
        pred = self.head(encoded).permute(0, 2, 1)
        pred = pred * scale.unsqueeze(1) + mean.unsqueeze(1)
        f_dim = -1 if self.configs.features == "MS" else 0
        return {
            "pred": pred[:, -y.size(1) :, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:],
            "support_density": support[..., 0].detach(),
            "support_recency": support[..., 2].detach(),
        }
