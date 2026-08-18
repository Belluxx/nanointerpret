# Experiments

> [!CAUTION]
> Using `--cache-activations` saves LLM activations to disk for reuse across runs. The default fp16 cache takes up a LOT of space (about 1.3GB every 1M tokens for the 270m Gemma model). `--residual-cache-format int8` takes about half as much space.

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

Disabling gradient clipping slightly improved validation metrics and increased training throughput by 35%.

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
| ✗ | **0.00236** | **99.44%** | **0.14%** | **85k tokens/s** |

## Fixing Qwen first-token activation outliers

Qwen tends to have extremely large residual-stream activations at the first sequence token ([paper](https://arxiv.org/pdf/2605.11887), bottom of page 2).

To prevent them from dominating SAE normalization / training, a raw L2-norm filter was added (`--max-activation-l2`).

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
