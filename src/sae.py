import math

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
        *,
        sink_norm_threshold: float | None,
        passthrough_dims: list[int],
        activation_scale: float,
        subtract_pre_bias: bool,
    ):
        super().__init__()
        keep = torch.ones(d_model, dtype=torch.bool)
        keep[passthrough_dims] = False
        self.register_buffer("keep", keep, persistent=False)
        d_in = int(keep.sum())
        decoder = F.normalize(torch.randn(d_sae, d_in), dim=1)
        self.encoder_weight = nn.Parameter(decoder.T.contiguous())
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae))
        self.decoder_weight = nn.Parameter(decoder)
        self.decoder_bias = nn.Parameter(torch.zeros(d_in))
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.sink_norm_threshold = math.inf if sink_norm_threshold is None else sink_norm_threshold
        self.activation_scale = activation_scale
        self.subtract_pre_bias = subtract_pre_bias

    def is_sink(self, residual: Tensor) -> Tensor:
        # Attention-sink tokens bypass the SAE unchanged.
        return torch.linalg.vector_norm(residual, dim=-1, dtype=torch.float32) > self.sink_norm_threshold

    def normalize(self, residual: Tensor) -> Tensor:
        # (tokens, d_model) residuals -> scaled SAE inputs without the pass-through dims.
        return residual[:, self.keep].float() * self.activation_scale

    def inputs(self, residual: Tensor) -> Tensor:
        # (tokens, d_model) residuals -> SAE inputs of the non-sink tokens.
        return self.normalize(residual[~self.is_sink(residual)])

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pre_activations = x @ self.encoder_weight + self.encoder_bias
        if self.subtract_pre_bias:
            # Equal to encoding x - decoder_bias, but its gradient skips a batch-sized matmul.
            pre_activations = pre_activations - self.decoder_bias @ self.encoder_weight
        values, indices = torch.topk(F.relu(pre_activations), self.k, dim=-1, sorted=False)
        return indices, values, pre_activations

    def decode(self, indices: Tensor, values: Tensor) -> Tensor:
        return F.embedding_bag(indices, self.decoder_weight, mode="sum", per_sample_weights=values)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        indices, values, pre_activations = self.encode(x)
        return self.decode(indices, values) + self.decoder_bias, indices, values, pre_activations

    @torch.no_grad()
    def reconstruct(self, residual: Tensor) -> Tensor:
        # Swap the SAE-owned dims of raw (..., d_model) non-sink residuals for their reconstruction.
        flat = residual.reshape(-1, residual.shape[-1])
        kept = ~self.is_sink(flat)
        reconstruction = flat[kept]
        reconstruction[:, self.keep] = (self(self.normalize(reconstruction))[0] / self.activation_scale).to(flat.dtype)
        output = flat.clone()
        output[kept] = reconstruction
        return output.reshape_as(residual)

    @torch.no_grad()
    def init_pre_bias(self, residual: Tensor) -> None:
        self.decoder_bias.copy_(geometric_median(self.inputs(residual)))

    @torch.no_grad()
    def constrain_decoder_gradient(self) -> None:
        gradient = self.decoder_weight.grad
        gradient.sub_((gradient * self.decoder_weight).sum(dim=1, keepdim=True) * self.decoder_weight)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder_weight.div_(self.decoder_weight.norm(dim=1, keepdim=True).clamp_min_(1e-12))


def geometric_median(points: Tensor, max_iterations: int = 100, tolerance: float = 1e-5) -> Tensor:
    estimate = points.mean(dim=0)
    for _ in range(max_iterations):
        weights = torch.linalg.vector_norm(points - estimate, dim=1).clamp_min(1e-7).reciprocal()
        updated = (points * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        if torch.linalg.vector_norm(updated - estimate) <= tolerance:
            return updated
        estimate = updated
    return estimate


def auxk_loss(sae: TopKSAE, pre_activations: Tensor, error: Tensor, dead: Tensor, aux_k: int) -> Tensor:
    dead_pre_activations = pre_activations[:, dead]
    values, indices = torch.topk(dead_pre_activations, min(aux_k, len(dead)), dim=-1, sorted=False)
    values = F.relu(values)
    if pre_activations.device.type == "mps":
        # MPS dense matmul is much faster than embedding_bag at AuxK's large k.
        activations = torch.zeros_like(dead_pre_activations).scatter_(1, indices, values)
        reconstruction = activations @ sae.decoder_weight[dead]
    else:
        reconstruction = sae.decode(dead[indices], values)
    target = error.detach()
    if sae.subtract_pre_bias:
        # As in OpenAI's recipe: the shift cancels in the loss but routes AuxK's gradient into the pre-bias.
        reconstruction = reconstruction + sae.decoder_bias
        target = target + sae.decoder_bias.detach()
    variance = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target), target)
    return torch.nan_to_num(F.mse_loss(reconstruction, target) / variance, nan=0.0, posinf=0.0, neginf=0.0)


class RunningMetrics:
    def __init__(self, sae: TopKSAE):
        device = sae.decoder_bias.device
        self.count = 0
        self.x_sum = torch.zeros(sae.d_in, device=device)
        self.x_sq_sum = torch.zeros((), device=device)
        self.error_sq_sum = torch.zeros((), device=device)
        self.fire_counts = torch.zeros(sae.d_sae, device=device)
        self.auxk_sum = torch.zeros((), device=device)
        self.auxk_count = 0

    @torch.no_grad()
    def update(
        self, x: Tensor, reconstruction: Tensor, indices: Tensor, values: Tensor, auxk: Tensor | None = None
    ) -> None:
        self.count += len(x)
        self.x_sum += x.sum(dim=0)
        self.x_sq_sum += x.square().sum()
        self.error_sq_sum += (x - reconstruction).square().sum()
        fired = indices[values > FIRING_THRESHOLD]
        self.fire_counts.scatter_add_(0, fired, torch.ones_like(fired, dtype=torch.float32))
        if auxk is not None:
            self.auxk_sum += auxk.detach() * len(x)
            self.auxk_count += len(x)

    def compute(self) -> dict[str, float | None]:
        variance = self.x_sq_sum - self.x_sum.square().sum() / self.count
        return {
            "mse": (self.error_sq_sum / (self.count * len(self.x_sum))).item(),
            "explained_variance": (1 - self.error_sq_sum / variance).item(),
            "auxk_loss": (self.auxk_sum / self.auxk_count).item() if self.auxk_count else None,
        }
