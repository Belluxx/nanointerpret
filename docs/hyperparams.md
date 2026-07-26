- `activation layer`: At which point of the LLM we extract the residual-stream vector.
    - Early layers are about lexical/syntactic features
    - Middle layers often give useful conceptual features
    - Late layers are usually prediction-oriented (not useful for interpretability)
- `width multiplier`: Number of SAE features. High multiplier separates concepts better but needs more data and leaves more dead features behind otherwise.
- `K`: Maximum features active per token.
    - Low values can be more interpretable but lead to features that are too broad.
    - Higher values are more specific, but may cause duplicates or "dirty" interpretations.
- `AuxK`: Number of dead features used to reconstruct the error left by the primary TopK representation. It defaults to the nearest power of two to half the residual width (`256` for the default `d_model=640`).
    - The normalized auxiliary loss has coefficient `1/32`.
    - A feature is considered to fire above activation `1e-3`.
    - Features that have not fired in the last 10,000,000 tokens are eligible for AuxK.

## Methodology

- **Architecture:** Top-K SAE with a learned encoder bias and decoder bias. The encoder receives the globally scaled activation directly; there is no pre-encoder decoder-bias subtraction. Both biases are initialized to zero.
  Sources: Gao et al., [*Scaling and evaluating sparse autoencoders*](https://arxiv.org/abs/2406.04093) for Top-K sparsity; Anthropic, [*Update on how we train SAEs*](https://transformer-circuits.pub/2024/april-update/index.html) for direct encoder input and zero-initialized biases.
- **Dead-feature prevention:** AuxK selects dead latents after updating firing timestamps from the primary TopK activations. These latents reconstruct a detached copy of the primary residual error with per-batch normalized MSE. AuxK never contributes to primary sparsity or feature-density metrics.
- **Input scaling:** One dataset-level scalar is estimated from a sample of training activations so that their average squared L2 norm equals the residual stream dimension. The same scalar is used for training and validation. Activations are not normalized independently per token.
  Source: Anthropic, [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
