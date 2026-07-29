# Experiments

## Pre-bias subtraction and AuxK are both useful

All runs used a Top-K SAE with `K=16`, width multiplier `16`, 300M training tokens, and 10M validation tokens.

<details>
<summary>Commands</summary>

```sh
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --gradient-clip 1 --output-dir artifacts/300M_aux_sub
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --gradient-clip 1 --output-dir artifacts/300M_sub --aux-k-coef 0
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --gradient-clip 1 --output-dir artifacts/300M_aux --no-subtract-pre-bias
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --gradient-clip 1 --output-dir artifacts/300M_plain --no-subtract-pre-bias --aux-k-coef 0
```

</details>

| Pre-bias subtraction | AuxK | Validation MSE | Explained variance | Dead features |
|:---:|:---:|---:|---:|---:|
| ✓ | ✓ | **0.00239** | **99.43%** | **0.1%** |
| ✓ | ✗ | 0.00251 | 99.40% | 2.1% |
| ✗ | ✓ | 0.00248 | 99.41% | 14.82% |
| ✗ | ✗ | 0.00403 | 99.04% | 91.88% |

## Gradient clipping is unnecessary

Disabling gradient clipping had no downsides and training throughput by 17%.

<details>
<summary>Commands</summary>

```sh
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --gradient-clip 1 --output-dir artifacts/300M_aux_sub
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux_sub_noclip
```

</details>

| Gradient clipping | Validation MSE | Explained variance | Dead features | Training throughput |
|:---:|---:|---:|---:|---:|
| ✓ (1) | 0.00239 | 99.43% | 0.1% | 26k tokens/s |
| ✗ | **0.00238** | 99.43% | 0.1% | **31k tokens/s** |
