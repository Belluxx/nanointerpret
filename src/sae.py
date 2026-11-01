import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FIRING_THRESHOLD = 1e-3


def matmul_dtype(tensor: Tensor) -> torch.dtype:
    # On CUDA the SAE's large matmuls take bf16 inputs and give float32 outputs: about twice as fast
    # as TF32, while the top-k still ranks float32 pre-activations.
    return torch.bfloat16 if tensor.is_cuda else tensor.dtype


def matmul(a: Tensor, b: Tensor, bias: Tensor | None = None) -> Tensor:
    # Float32 bias + a @ b, with inputs in the matmul dtype.
    if not a.is_cuda:
        return a @ b if bias is None else torch.addmm(bias, a, b)
    a, b = a.bfloat16(), b.bfloat16()
    return torch.mm(a, b, out_dtype=torch.float32) if bias is None else torch.addmm(bias, a, b, out_dtype=torch.float32)


class SelectPreActivations(torch.autograd.Function):
    # The pre-activations (x - shift) @ weight + bias at `indices`, gathered from their precomputed
    # values. Only these entries get gradients, so the backward pass builds the dense gradient in the
    # matmul dtype and sums the bias gradient over them alone.
    @staticmethod
    def forward(ctx, pre_activations: Tensor, indices: Tensor, x: Tensor, weight: Tensor, bias: Tensor, shift: Tensor) -> Tensor:
        ctx.save_for_backward(indices, x, weight, shift)
        return pre_activations.gather(1, indices)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[None, None, None, Tensor, Tensor, Tensor | None]:
        indices, x, weight, shift = ctx.saved_tensors
        dtype = matmul_dtype(grad)
        dense_grad = grad.new_zeros(len(grad), weight.shape[1], dtype=dtype).scatter_add_(1, indices, grad.to(dtype))
        grad_bias = grad.new_zeros(weight.shape[1]).index_add_(0, indices.flatten(), grad.flatten())
        # The shift's gradient follows from the bias's, skipping a batch-sized matmul.
        grad_shift = -(weight @ grad_bias) if ctx.needs_input_grad[5] else None
        return None, None, None, matmul((x - shift).T, dense_grad), grad_bias, grad_shift


class Decode(torch.autograd.Function):
    # embedding_bag's sums of value-weighted decoder rows, with a backward pass of a gather and two
    # matmuls: a third of the cost of embedding_bag's on CUDA, and far less on MPS.
    @staticmethod
    def forward(ctx, indices: Tensor, values: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(indices, values, weight)
        return F.embedding_bag(indices, weight, mode="sum", per_sample_weights=values)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[None, Tensor, Tensor]:
        indices, values, weight = ctx.saved_tensors
        dtype = matmul_dtype(grad)
        grad_values = (F.embedding(indices, weight.to(dtype)) @ grad.to(dtype).unsqueeze(2)).squeeze(2).to(grad.dtype)
        activations = grad.new_zeros(len(grad), len(weight), dtype=dtype).scatter_(1, indices, values.to(dtype))
        return None, grad_values, matmul(activations.T, grad)


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

    def shift(self) -> Tensor:
        # Subtracted from inputs before encoding.
        return self.decoder_bias if self.subtract_pre_bias else torch.zeros_like(self.decoder_bias)

    @torch.no_grad()
    def pre_activations(self, x: Tensor) -> Tensor:
        # Without gradients, which only the selected features get, through `select`.
        return matmul(x - self.shift(), self.encoder_weight, self.encoder_bias)

    def top_k(self, pre_activations: Tensor) -> Tensor:
        return torch.topk(pre_activations, self.k, dim=-1, sorted=False).indices

    def select(self, x: Tensor, pre_activations: Tensor, indices: Tensor) -> Tensor:
        # Activations at `indices`. The ReLU comes after the top-k, as it keeps the order.
        return F.relu(SelectPreActivations.apply(pre_activations, indices, x, self.encoder_weight, self.encoder_bias, self.shift()))

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pre_activations = self.pre_activations(x)
        indices = self.top_k(pre_activations)
        return indices, self.select(x, pre_activations, indices)

    def decode(self, indices: Tensor, values: Tensor) -> Tensor:
        return Decode.apply(indices, values, self.decoder_weight) + self.decoder_bias

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        indices, values = self.encode(x)
        return self.decode(indices, values), indices, values

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
        parallel = torch.linalg.vecdot(gradient, self.decoder_weight).unsqueeze(1)
        gradient.addcmul_(parallel, self.decoder_weight, value=-1)

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


def auxk_loss(sae: TopKSAE, dead: Tensor, indices: Tensor, values: Tensor, error: Tensor) -> Tensor:
    # `values` activate the dead features `dead[indices]`. At AuxK's large k a dense matmul is much
    # faster than embedding_bag.
    activations = values.new_zeros(len(values), len(dead)).scatter_(1, indices, values)
    reconstruction = activations @ sae.decoder_weight[dead]
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
        self.fire_counts.scatter_add_(0, indices.flatten(), (values > FIRING_THRESHOLD).flatten().float())
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
