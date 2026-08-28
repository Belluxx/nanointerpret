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
    - A feature is considered to fire above activation `1e-3`.
- `aux_k_coef`: Weight on the normalized AuxK reconstruction loss. The default `1/32` is the value used by OpenAI for TopK SAEs. [1]
    - Larger values put more optimization pressure on dead features to explain the primary reconstruction error; smaller values make AuxK less influential. `0` disables AuxK. [1]
- `dead_window`: Number of tokens a feature can go without firing before it becomes dead and eligible for AuxK. The default is 10M tokens.
- `model_batch_size`: contexts processed by the language model together. The default is `32`.
- `sae_batch_size`: residual-stream token vectors count. The default is `4096`; this is an optimization batch, not just a data-loading setting. OpenAI used much larger batches for parallelism but the converged loss was not strongly batch-dependent. [1]
- `max_activation_l2`: Optional raw residual-stream L2 cutoff (`--max-activation-l2`). Use `auto` to automatically detect the threshold. This was necessary only for Qwen models (so far) because they tend to have extremely high activations for the first token.
- `learning_rate`: By default it is automatically calculated with `3e-4 * sqrt(32768 / d_sae)`. It s a good heuristic based on initial experiments and OpenAI research. [1]

## Methodology

- This project combines Anthropic's activation setup [2] with Gao et al.'s Top-K SAE [1].
- By default, training streams activations into the SAE, without writing a residual cache, and keeps the LLM loaded. `--cache-activations` stores them as fp16 by default. Pass `--residual-cache-format int8` to use about half the space. Caching activations is very useful when doing ablation tests, as you avoid recalculating the same activations for each test.
- On MPS, LLM layers are compiled for faster activations extraction. Pass `--no-compile-model` to disable it.
- By default, activations come from the input to the middle transformer layer. A single scale is applied so their average squared L2 norm equals the residual width. [2]
- The SAE uses Top-K sparsification, tied encoder/decoder initialization, a shared geometric-median bias, unit-norm decoder directions, and AuxK. AuxK helps revive features that have not fired after many tokens. [1]
- Gradient clipping is disabled by default after [experiments found no benefit](experiments.md#gradient-clipping-is-unnecessary).
- Periodic evaluation measures mean `KL(base_logits || sae_logits)`. Lower KL is the primary model-preservation metric and logged in `evaluation_metrics.jsonl`.

Sources:
- [1] [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)
- [2] [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
