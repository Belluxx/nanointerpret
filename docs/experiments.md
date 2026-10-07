# Experiments

## Lessons learned

- Lowering `K` from `32` to `16` improves interpretability (but costs reconstruction accuracy). Increasing SAE width from `16x` to `32x` sometimes separates fused concepts better, but needs more data and (in Gemma 270M) doesn't fix its poor text understanding due to its tiny size. It also significantly increases duplicate features.
- A later-layer Gemma run produced more abstract features, but they became incoherent already at medium activation levels. It's important to check the whole activation distribution.
- FineWeb-Edu was a poor choice, features seemed to encode the dataset's educational style. For example, an apparently general fire-related feature steered the model toward educational fire-themed completions. This is evident for many features in both Gemma3 270M and Qwen3 1.7B.

Hardware: M4 Max Mac Studio (CPU 16C, GPU 40C, 64GB RAM)

## MPS streaming performance
On Apple Silicon compiling each LLM layer improves its throughput by ~100% for Gemma3 270M, ~50% for Qwen3 0.6B, ~8% for Qwen3 1.7B.

<details>
<summary>Command</summary>

```sh
python train.py --model-id unsloth/Qwen3-0.6B-Base --activation-layer 14 --train-tokens 1000000 --validation-tokens 100000 --normalization-tokens 100000 --dead-window 100000 --model-batch-size 32 --sae-batch-size 4096 --width-multiplier 16 --k 16
python train.py --model-id unsloth/Qwen3-0.6B-Base --activation-layer 14 --train-tokens 1000000 --validation-tokens 100000 --normalization-tokens 100000 --dead-window 100000 --model-batch-size 32 --sae-batch-size 4096 --width-multiplier 16 --k 16 --no-compile-model
python train.py --model-id unsloth/Qwen3-1.7B-Base --activation-layer 14 --train-tokens 1000000 --validation-tokens 100000 --normalization-tokens 100000 --dead-window 100000 --model-batch-size 32 --sae-batch-size 4096 --width-multiplier 16 --k 16
python train.py --model-id unsloth/Qwen3-1.7B-Base --activation-layer 14 --train-tokens 1000000 --validation-tokens 100000 --normalization-tokens 100000 --dead-window 100000 --model-batch-size 32 --sae-batch-size 4096 --width-multiplier 16 --k 16 --no-compile-model
```

</details>

| Model | Prefix compilation | Activation capture | SAE training | Streamed training |
|---|:---:|---:|---:|---:|
| Qwen3-0.6B, layer 14 | ✗ | 18.35k tok/s | 35.47k tok/s | 12.12k tok/s |
| Qwen3-0.6B, layer 14 | ✓ | **28.54k tok/s** | 35.22k tok/s | **15.89k tok/s (+31%)** |
| Qwen3-1.7B, layer 14 | ✗ | 7.89k tok/s | 17.41k tok/s | 5.41k tok/s |
| Qwen3-1.7B, layer 14 | ✓ | **9.70k tok/s** | 17.39k tok/s | **6.18k tok/s (+14%)** |

## Pre-bias subtraction and AuxK

Pre-bias subtraction and AuxK were both useful for training SAEs.

<details>
<summary>Commands</summary>

```sh
python3 train.py --model-id unsloth/gemma-3-270m --dataset-id HuggingFaceFW/fineweb-edu --k 16 --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux_sub
python3 train.py --model-id unsloth/gemma-3-270m --dataset-id HuggingFaceFW/fineweb-edu --k 16 --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_sub --aux-k-coef 0
python3 train.py --model-id unsloth/gemma-3-270m --dataset-id HuggingFaceFW/fineweb-edu --k 16 --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux --no-subtract-pre-bias
python3 train.py --model-id unsloth/gemma-3-270m --dataset-id HuggingFaceFW/fineweb-edu --k 16 --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_plain --no-subtract-pre-bias --aux-k-coef 0
```

</details>

| Pre-bias subtraction | AuxK | Validation MSE | Explained variance | Dead features |
|:---:|:---:|---:|---:|---:|
| ✓ | ✓ | **0.002327** | **99.444%** | **0.107%** |
| ✓ | ✗ | 0.002485 | 99.407% | 23.779% |
| ✗ | ✓ | 0.002549 | 99.391% | 21.270% |
| ✗ | ✗ | 0.004399 | 98.949% | 94.570% |

## Gradient clipping is unnecessary

Disabling gradient clipping slightly improved validation metrics and increased training throughput by ~40%.

<details>
<summary>Commands</summary>

```sh
python3 train.py --model-id unsloth/gemma-3-270m --dataset-id HuggingFaceFW/fineweb-edu --k 16 --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux_sub
python3 train.py --model-id unsloth/gemma-3-270m --dataset-id HuggingFaceFW/fineweb-edu --k 16 --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux_sub_clip --gradient-clip 1
```

</details>

| Gradient clipping | Validation MSE | Explained variance | Dead features | Training throughput |
|:---:|---:|---:|---:|---:|
| ✓ (1) | 0.002335 | 99.442% | 0.127% | 65k tokens/s |
| ✗ | **0.002327** | **99.444%** | **0.107%** | **91k tokens/s (+40%)** |

## Passing through massive-activation dimensions

Both Qwen and Gemma have a few residual-stream dimensions with extremely large activations. In Qwen they spike at the first token of every context ([paper](https://arxiv.org/pdf/2605.11887), bottom of page 2), in Gemma at BOS and some punctuation.

They carry very little information (basically how much a token acts as an attention sink), but they dominate SAE normalization / training and inflate the metrics. In Gemma 3 270M a single dimension holds 96% of the variance, so explained variance looks great even when the SAE is not.

To fix this, dimensions whose std is more than 15x the median dimension std are passed through: the SAE ignores them and they keep their original value when the reconstruction is fed back into the model. It adapts to each model automatically and it's on by default, use `--no-passthrough-massive-dims` to disable it.

In Qwen these dimensions only spike on attention-sink tokens, which are now [excluded](#excluding-attention-sink-tokens) before the std is measured, so no dimension is passed through there. Gemma still needs it.

![Massive-activation dimensions and token norms for Gemma and Qwen](../assets/plots/massive_activation_dims.png)

## Excluding attention-sink tokens

In Qwen, massive activations belong to a few tokens rather than to dimensions. Sink tokens have a residual norm 65-92x the median token's and every other token stays under 1.7x, with nothing in between (131k training tokens, layer 14). They are the first token of every context (Qwen has no BOS, so it's an arbitrary mid-document token) and rarely a token at position 1-2. Gemma has no such gap: its BOS and punctuation tokens reach at most 16.5x, and its massive dimensions stay massive over the remaining tokens.

Sinks waste SAE capacity. The previous Qwen 1.7B layer 14 SAE (k=32, 100M tokens) dedicated 39 features almost only to position 0 (over 90% of their firings), taking 93.6% of the position-0 firings, and their recorded examples are arbitrary tokens.

Tokens whose residual norm exceeds 30x the median one (between Gemma's maximum and Qwen's sinks) are now left out of calibration, training, validation, and recording, and keep their original residual in the downstream KL. Calibration then finds no massive dimensions in Qwen, so the SAE models all of them.

<details>
<summary>Commands</summary>

```sh
python3 train.py --model-id unsloth/Qwen3-1.7B-Base --activation-layer 14 --dataset-id HuggingFaceFW/fineweb --train-tokens 1048576 --validation-tokens 131072 --recording-tokens 262144
python3 train.py --model-id unsloth/gemma-3-270m --activation-layer 9 --dataset-id HuggingFaceFW/fineweb --train-tokens 2000000 --validation-tokens 262144 --recording-tokens 1048576
```

</details>

| Model | Sink tokens (1M calibration tokens) | Pass-through dims |
|---|---:|---|
| Qwen3 1.7B, layer 14 | 0.39% | none (previously 1401, 1793, 1999) |
| Gemma 3 270M, layer 9 | 0% | 163, 400 (unchanged) |

It's on by default, use `--no-exclude-sinks` to disable it.
