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
- `exclude_sinks`: Attention-sink tokens, whose residual norm is over 30x the median token's, bypass the SAE: they are left out of training, validation, and recording, and keep their original residual when the reconstruction is fed back into the model (`--no-exclude-sinks` to disable). In Qwen they are the first token of every context plus a few rare ones, Gemma has none ([details](experiments.md#massive-activations)).
- `passthrough_massive_dims`: Residual dimensions whose std over non-sink tokens is over 15x the median one are passed through, so the SAE ignores them (`--no-passthrough-massive-dims` to disable). Needed for Gemma because a few dimensions have extremely large activations across many tokens; Qwen has none once sink tokens are excluded ([details](experiments.md#massive-activations)).
- `learning_rate`: By default it is automatically calculated with `3e-4 * sqrt(32768 / d_sae)`. It s a good heuristic based on initial experiments and OpenAI research. [1]

## Methodology

- This project combines Anthropic's activation setup [2] with Gao et al.'s Top-K SAE [1].
- By default, training streams activations into the SAE, without writing a residual cache, and keeps the LLM loaded. `--cache-activations` stores them as fp16 instead (2 bytes per value, so about 140GB for 100M Gemma 270M tokens). Caching activations is very useful when doing ablation tests, as you avoid recalculating the same activations for each test.
- On MPS, LLM layers are compiled for faster activations extraction, in both training and recording. Pass `--no-compile-model` to disable it.
- By default, activations come from the input to the middle transformer layer. A single scale is applied so the average squared L2 norm of non-sink tokens equals the SAE input width (residual width minus pass-through dims). [2]
- The SAE uses Top-K sparsification, tied encoder/decoder initialization, a shared geometric-median bias, unit-norm decoder directions, and AuxK. AuxK helps revive features that have not fired after many tokens. [1]
- Gradient clipping is disabled by default after [experiments found no benefit](experiments.md#ablations).
- Periodic evaluation measures mean `KL(base_logits || sae_logits)`. Lower KL is the primary model-preservation metric and logged in `evaluation_metrics.jsonl`.

## Recorded activations

`record_activations.py` writes every firing activation of the recording split to `<sae-dir>/activations/`, indexed both ways:
- `token_ptr.npy`, `token_features.npy`, `token_values.npy`: the features and values of token `t` are the `token_ptr[t]:token_ptr[t + 1]` slices.
- `feature_ptr.npy`, `feature_positions.npy`, `feature_values.npy`: the token positions and values of feature `f`, in token order, are the `feature_ptr[f]:feature_ptr[f + 1]` slices.
- `token_ids.npy`: the recorded tokens. Position `p` is token `p % context_size` of its context; the model saw nothing before the context start, which may be mid-document.
- `sink_positions.npy`: sink tokens, which bypass the SAE and so have no features.
- `feature_max.npy` and `metadata.json` (SAE config, split offset, and firing threshold).

Values are fp16 in SAE input units; divide them by `activation_scale` for residual-stream units.

## Feature interpretation

`interpret.py` names each feature with an LLM, then checks the name ([results](experiments.md#feature-interpretation)):
- **Explain**: the explainer sees the 8 strongest activations and 12 drawn evenly from activation-quantile bins (0-50, 50-80, 80-95, 95-100%), 32 tokens before and 8 after, with the firing tokens marked and their activations on a 1-10 scale, plus the tokens the feature fires on most and the share of their occurrences it fires on. Sampling across the range keeps the name from describing only the peak. [3]
- **Detection**: a judge given only the name says whether the feature fires on each of 24 marked tokens it hasn't seen: 12 activations from the same bins, and 12 silent tokens split between random ones, the feature's frequent tokens where it stays silent, and activations of the feature with the closest decoder direction. The last two are hard negatives: a vague or token-level name says yes to them. Balanced accuracy averages recall with the specificity on each kind of silent token. [3]
- **Generated tests**: the judge writes 6 pairs of texts from the name, one meant to make the feature fire and a minimal contrast meant to keep it silent, and the model and SAE measure the actual activations; a pair passes when the contrast's activation is under half the first text's (a smaller drop is within what irrelevant changes cause). The model, not a judge, is the ground truth, so this catches names that are right about the examples but too broad. [4, 5]
- **Revision**: when fewer than 4 of 6 pairs pass, the explainer sees the measured activations and revises the name once; the version with the higher detection plus pass rate is kept. [5]
- **Steering**: the decoder direction is added at the SAE's layer on 8 neutral prompts, at the strongest multiple of the max activation (0.5-16x) that keeps the KL from the unsteered next-token predictions within 1 nat, and the tokens whose log-probability rises most are reported. A fixed multiple doesn't work: at 4x the KL ranges from under 0.5 to 6 nats across features. This stays separate from the name, since what a mid-layer feature fires on and what it promotes often differ. [6, 7]

Sources:
- [1] [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)
- [2] [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [3] [Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/abs/2410.13928)
- [4] [FADE: Why Bad Descriptions Happen to Good Features](https://arxiv.org/abs/2502.16994)
- [5] [SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization](https://arxiv.org/abs/2606.08496)
- [6] [SAEs Are Good for Steering – If You Select the Right Features](https://arxiv.org/abs/2505.20063)
- [7] [Enhancing Automated Interpretability with Output-Centric Feature Descriptions](https://arxiv.org/abs/2501.08319)
