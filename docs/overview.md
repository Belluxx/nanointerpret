# Overview

## Hyperparams
- `activation layer`: At which point of the LLM we extract the residual-stream vector.
    - Early layers are about lexical/syntactic features
    - Middle layers often give useful conceptual features
    - Late layers may be too prediction-oriented, but not always. (testing residual-stream extraction from `n-1` or a bit earlier may be interesting for abstract concepts that are less tied to token patterns)
- `width multiplier`: Controls the number of SAE features (`n_features = d_model * mult`). High multiplier separates concepts better but needs more data and leaves more dead features behind otherwise (or cause feature splitting).
- `K`: Maximum features active per token.
    - Low values can be more interpretable but lead to features that are too broad (worse reconstruction).
    - Higher values tend to be less interpretable (better reconstruction).
- `AuxK`: Number of dead features used to reconstruct the error left by the primary TopK representation. It defaults to a power of 2 close to half the residual width (`256` for the default `d_model=640`).
    - The normalized auxiliary loss has coefficient `1/32`.
    - A feature is considered to fire above activation `1e-3`.
    - Features that have not fired in the last 10M tokens are eligible for AuxK.

## Methodology

- Architecture: Top-K sparsification follows OpenAI [1], while the encoder/bias formulation follows Anthropic's updated SAE recipe [2] where the globally scaled activation is passed directly to the encoder without subtracting the decoder bias, and both biases are initialized to zero.
- Dead-feature prevention: AuxK selects dead latents after updating firing timestamps from the primary TopK activations. These latents reconstruct a detached copy of the primary residual error with per-batch normalized MSE. AuxK never contributes to primary sparsity or feature-density metrics. [1]
- Input scaling: One dataset-level scalar is estimated from a sample of training activations so that their average squared L2 norm equals the residual stream dimension. The same scalar is used for training and validation. Activations are not normalized independently per token. [3]

Sources:
- [1] [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)
- [2] [Update on how we train SAEs](https://transformer-circuits.pub/2024/april-update/index.html)
- [3] [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)