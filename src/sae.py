from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int, device: torch.device):
        super().__init__()
        if not 0 < k <= d_sae:
            raise ValueError(f"k must be in [1, {d_sae}], got {k}")

        decoder = F.normalize(torch.randn(d_sae, d_model, device=device), dim=1)
        self.encoder_weight = nn.Parameter(decoder.T.contiguous())
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae, device=device))
        self.decoder_weight = nn.Parameter(decoder)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model, device=device))
        self.k = k
        self.d_model = d_model
        self.d_sae = d_sae

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pre_activations = (x - self.decoder_bias) @ self.encoder_weight + self.encoder_bias
        values, indices = torch.topk(F.relu(pre_activations), self.k, dim=-1, sorted=False)
        return indices, values

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        indices, values = self.encode(x)
        reconstruction = F.embedding_bag(
            indices,
            self.decoder_weight,
            mode="sum",
            per_sample_weights=values,
        ) + self.decoder_bias
        return reconstruction, indices, values

    @torch.no_grad()
    def constrain_decoder_gradient(self) -> None:
        if self.decoder_weight.grad is None:
            return
        parallel = (self.decoder_weight.grad * self.decoder_weight).sum(dim=1, keepdim=True)
        self.decoder_weight.grad.sub_(parallel * self.decoder_weight)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.decoder_weight.norm(dim=1, keepdim=True).clamp_min_(1e-12)
        self.decoder_weight.div_(norms)


class RunningMetrics:
    """Accumulate reconstruction and feature-use metrics for one logging window."""

    RARE_FREQUENCY = 1e-4
    OVERACTIVE_FREQUENCY = 1e-2

    def __init__(self, d_model: int, d_sae: int, device: torch.device):
        self.device = device
        self.d_model = d_model
        self.d_sae = d_sae
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.l0_sum = torch.zeros((), device=self.device)
        self.x_sum = torch.zeros(self.d_model, device=self.device)
        self.x_sq_sum = torch.zeros(self.d_model, device=self.device)
        self.error_sum = torch.zeros(self.d_model, device=self.device)
        self.error_sq_sum = torch.zeros(self.d_model, device=self.device)
        self.feature_fire_counts = torch.zeros(self.d_sae, device=self.device)

    @torch.no_grad()
    def update(
        self, x: Tensor, reconstruction: Tensor, indices: Tensor, values: Tensor
    ) -> Tensor:
        error = x - reconstruction
        self.count += x.shape[0]
        self.l0_sum += (values > 0).sum()
        self.x_sum += x.sum(dim=0)
        self.x_sq_sum += x.square().sum(dim=0)
        self.error_sum += error.sum(dim=0)
        self.error_sq_sum += error.square().sum(dim=0)
        positive_indices = indices[values > 0]
        batch_fire_counts = torch.zeros_like(self.feature_fire_counts)
        batch_fire_counts.scatter_add_(
            0, positive_indices, torch.ones_like(positive_indices, dtype=torch.float32)
        )
        self.feature_fire_counts += batch_fire_counts
        return batch_fire_counts

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {"mse": math.nan, "explained_variance": math.nan, "l0": math.nan}

        n = float(self.count)
        sse = self.error_sq_sum.sum()
        x_variance = (self.x_sq_sum - self.x_sum.square() / n).sum().clamp_min(1e-12)
        error_variance = (self.error_sq_sum - self.error_sum.square() / n).sum().clamp_min(0)
        frequencies = self.feature_fire_counts.cpu().numpy() / n
        active = frequencies > 0
        return {
            "mse": float((sse / (n * self.d_model)).item()),
            "explained_variance": float((1.0 - error_variance / x_variance).item()),
            "l0": float((self.l0_sum / n).item()),
            "window_dead_feature_pct": float(100.0 * (~active).mean()),
            "window_rare_feature_pct": float(
                100.0 * (active & (frequencies < self.RARE_FREQUENCY)).mean()
            ),
            "window_overactive_feature_pct": float(
                100.0 * (frequencies > self.OVERACTIVE_FREQUENCY).mean()
            ),
        }
