# Experiments

> [!CAUTION]
> Using the `--cache-activation` flag saves LLM activations to disk to reuse them across runs, however they take up a LOT of space (~1.3GB for each 1M tokens), be careful with it. If you don't have a lot of space, remove the flag.

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

It resulted in pretty good results, identifying interesting features like `#1009` which identified **severe harm** (*injured*, *killed*, *lives*, *wounded*, *deaths*, *dead*, *lost*, ...) and `#9147` which identified **professions** (*fisherman*, *photographer*, *painter*, *farmer*, *artist*, ...).

However many features were polysemantic, for example the severe harm feature activated both with causalities terms and with **employment terms** (*employment*, *workers*, gained and lost jobs).

My first 2 ideas to fix this issue were the following:
- Increase `--width-multiplier` (number of features in the SAE): this may help in separating fused features
- Extract the residual stream from a later layer like `13`-`15` instead of the middle one (`9`). Especially after seeing that many of the features activate on identical tokens, so moving to later layers should help with higher-abstraction representations.