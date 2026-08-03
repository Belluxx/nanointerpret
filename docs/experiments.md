# Experiments

> [!CAUTION]
> Using the `--cache-activation` flag makes it possible to save the LLM activations and reuse them during ablation tests, however they get saved to disk and take up a LOT of space (~1.3GB for each 1M tokens), be careful with it. If you don't have a lot of space, remove the flag.

## Pre-bias subtraction and AuxK are both useful

All runs used a Top-K SAE with `K=16`, width multiplier `16`, 300M training tokens, and 10M validation tokens. Gradient clipping was disabled unless stated otherwise.

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
