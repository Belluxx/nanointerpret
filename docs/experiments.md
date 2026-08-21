# Experiments

> [!CAUTION]
> Using `--cache-activations` saves LLM activations to disk for reuse across runs. The default fp16 cache takes up a LOT of space (about 1.3GB every 1M tokens for the 270m Gemma model). `--residual-cache-format int8` takes about half as much space.

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
python3 train.py --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux_sub
python3 train.py --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_sub --aux-k-coef 0
python3 train.py --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux --no-subtract-pre-bias
python3 train.py --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_plain --no-subtract-pre-bias --aux-k-coef 0
```

</details>

| Pre-bias subtraction | AuxK | Validation MSE | Explained variance | Dead features |
|:---:|:---:|---:|---:|---:|
| ✓ | ✓ | **0.00236** | **99.44%** | **0.14%** |
| ✓ | ✗ | 0.00244 | 99.42% | 6.88% |
| ✗ | ✓ | 0.00252 | 99.40% | 16.32% |
| ✗ | ✗ | 0.00406 | 99.03% | 91.56% |

## Gradient clipping is unnecessary

Disabling gradient clipping slightly improved validation metrics and increased training throughput by ~35%.

<details>
<summary>Commands</summary>

```sh
python3 train.py --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux_sub
python3 train.py --cache-activations --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux_sub_clip --gradient-clip 1
```

</details>

| Gradient clipping | Validation MSE | Explained variance | Dead features | Training throughput |
|:---:|---:|---:|---:|---:|
| ✓ (1) | 0.00239 | 99.43% | 0.18% | 63k tokens/s |
| ✗ | **0.00236** | **99.44%** | **0.14%** | **85k tokens/s (+35%)** |

## Fixing Qwen first-token activation outliers

Qwen tends to have extremely large residual-stream activations at the first sequence token ([paper](https://arxiv.org/pdf/2605.11887), bottom of page 2).

To prevent them from dominating SAE normalization / training, a raw L2-norm filter was added (`--max-activation-l2`).

Gemma 3 270M also has very large residual-stream activations, however they occur over tokens like BOS and punctuation. So they are more complex and potentially meaningful, unlike Qwen's case. I would not recommend L2 filtering for Gemma by default but feel free to test it.

## First stable training run

The first stable training run was the following:

```sh
python3 train.py \
  --output-dir artifacts/500M \
  --train-tokens 500000000 \
  --validation-tokens 100000000 \
  --checkpoint-every 250000000 \
  --activation-layer 9 \
  --width-multiplier 16 \
  --k 16 \
  --aux-k 256 \
  --aux-k-coef 0.03125 \
  --dead-window 10000000 \
  --learning-rate 0.0003 \
  --sae-batch-size 4096
```

It resulted in pretty good results, identifying interesting features:
- `#1009` identifies **severe harm** (*injured*, *killed*, *lives*, *wounded*, *deaths*, *dead*, *lost*, ...)
- `#9147` identifies **professions** (*fisherman*, *photographer*, *painter*, *farmer*, *artist*, ...)
- `#382` identifies **nutrient and ingested substances** very cleanly, even at lot activations (*flavonoids*, *carotenoid*, *lead*, *calcium*, *vitamin D3*, ...)

However many features were polysemantic, for example the severe harm feature activated both with causalities terms and with **employment terms** (*employment*, *workers*, gained and lost jobs).

My first 2 ideas to fix this issue were the following:
- Increase `--width-multiplier` (number of features in the SAE): this may help in separating fused features
- Extract the residual stream from a later layer like `13`-`15` instead of the middle one (`9`). Especially after seeing that many of the features activate on identical tokens, so moving to later layers should help with higher-abstraction representations.

## Run 2 (32x multiplier)

The second run only increased `--width-multiplier` from `16` to `32`.

```sh
python3 train.py \
  --output-dir artifacts/500M_w32 \
  --train-tokens 500000000 \
  --validation-tokens 100000000 \
  --checkpoint-every 250000000 \
  --activation-layer 9 \
  --width-multiplier 32 \
  --k 16 \
  --aux-k 256 \
  --aux-k-coef 0.03125 \
  --dead-window 10000000 \
  --learning-rate 0.0003 \
  --sae-batch-size 4096
```

The results were better. The feature `#1009` from the previous run (the one that mixed "severe harm" with "employment terms") was now correctly separated between `#19969` and `#13311`, where:
- `#19969` isolates "casualties and destructive outcomes"
- `#13311` isolates "workforce and human labour capacity"

Note that these two features are not perfectly clean, there still some overlap between them. Probably some of it is due to the fact that the LLM used is extremely small (`0.2B` params).

Other interesting features were:
- `#13180`: deception, misinformation, and betrayal
- `#8727`: kindness and compassion
- `#15023`: catastrophic or debilitating severity
