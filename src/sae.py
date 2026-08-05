from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


FIRING_THRESHOLD = 1e-3


class TopKSAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_sae: int,
        k: int,
        device: torch.device,
        *,
        subtract_pre_bias: bool = True,
    ):
        super().__init__()
        decoder = F.normalize(torch.randn(d_sae, d_model, device=device), dim=1)
        self.encoder_weight = nn.Parameter(decoder.T.contiguous())
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae, device=device))
        self.decoder_weight = nn.Parameter(decoder)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model, device=device))
        self.k = k
        self.d_model = d_model
        self.d_sae = d_sae
        self.subtract_pre_bias = subtract_pre_bias

    def decode(
        self, indices: Tensor, values: Tensor, *, include_bias: bool = True
    ) -> Tensor:
        reconstruction = F.embedding_bag(
            indices,
            self.decoder_weight,
            mode="sum",
            per_sample_weights=values,
        )
        if include_bias:
            reconstruction = reconstruction + self.decoder_bias
        return reconstruction

    def forward_with_pre_activations(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.subtract_pre_bias:
            x = x - self.decoder_bias
        pre_activations = x @ self.encoder_weight + self.encoder_bias
        values, indices = torch.topk(
            F.relu(pre_activations), self.k, dim=-1, sorted=False
        )
        reconstruction = self.decode(indices, values)
        return reconstruction, indices, values, pre_activations

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        reconstruction, indices, values, _pre_activations = (
            self.forward_with_pre_activations(x)
        )
        return reconstruction, indices, values

    @torch.no_grad()
    def constrain_decoder_gradient(self) -> None:
        gradient = self.decoder_weight.grad
        parallel = (gradient * self.decoder_weight).sum(dim=1, keepdim=True)
        gradient.sub_(parallel * self.decoder_weight)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.decoder_weight.norm(dim=1, keepdim=True).clamp_min_(1e-12)
        self.decoder_weight.div_(norms)


def normalized_auxk_loss(
    sae: TopKSAE,
    pre_activations: Tensor,
    residual_error: Tensor,
    dead_mask: Tensor,
    aux_k: int,
) -> Tensor:
    dead_count = int(dead_mask.sum().item())
    if dead_count == 0:
        return pre_activations.sum() * 0.0

    masked = pre_activations.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
    values, indices = torch.topk(
        masked, min(aux_k, dead_count), dim=-1, sorted=False
    )
    auxiliary_reconstruction = sae.decode(
        indices, F.relu(values), include_bias=sae.subtract_pre_bias
    )
    target = residual_error.detach()
    if sae.subtract_pre_bias:
        target = target + sae.decoder_bias.detach()
    numerator = F.mse_loss(auxiliary_reconstruction, target)
    target_mean = target.mean(dim=0, keepdim=True)
    denominator = F.mse_loss(target_mean.expand_as(target), target)
    return torch.nan_to_num(numerator / denominator, nan=0.0, posinf=0.0, neginf=0.0)


class RunningMetrics:
    def __init__(self, d_model: int, d_sae: int, device: torch.device):
        self.device = device
        self.d_model = d_model
        self.d_sae = d_sae
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.x_sum = torch.zeros(self.d_model, device=self.device)
        self.x_sq_sum = torch.zeros(self.d_model, device=self.device)
        self.error_sq_sum = torch.zeros(self.d_model, device=self.device)
        self.feature_fire_counts = torch.zeros(self.d_sae, device=self.device)
        self.auxk_loss_sum = torch.zeros((), device=self.device)
        self.auxk_token_count = 0

    @torch.no_grad()
    def update(
        self,
        x: Tensor,
        reconstruction: Tensor,
        indices: Tensor,
        values: Tensor,
        auxk_loss: Tensor | None = None,
    ) -> Tensor:
        error = x - reconstruction
        self.count += x.shape[0]
        firing = values > FIRING_THRESHOLD
        self.x_sum += x.sum(dim=0)
        self.x_sq_sum += x.square().sum(dim=0)
        self.error_sq_sum += error.square().sum(dim=0)
        positive_indices = indices[firing]
        batch_fire_counts = torch.zeros_like(self.feature_fire_counts)
        batch_fire_counts.scatter_add_(
            0, positive_indices, torch.ones_like(positive_indices, dtype=torch.float32)
        )
        self.feature_fire_counts += batch_fire_counts
        if auxk_loss is not None:
            self.auxk_loss_sum += auxk_loss.detach() * x.shape[0]
            self.auxk_token_count += x.shape[0]
        return batch_fire_counts

    @torch.no_grad()
    def compute(self) -> dict[str, float | None]:
        n = float(self.count)
        sse = self.error_sq_sum.sum()
        x_variance = (self.x_sq_sum - self.x_sum.square() / n).sum().clamp_min(1e-12)
        mse = float((sse / (n * self.d_model)).item())
        result = {
            "mse": mse,
            "explained_variance": float((1.0 - sse / x_variance).item()),
            "auxk_loss": None,
        }
        if self.auxk_token_count:
            result["auxk_loss"] = float(
                (self.auxk_loss_sum / self.auxk_token_count).item()
            )
        return result
