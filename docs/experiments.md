# Experiments

## Pre-bias subtraction and AuxK are both useful

All runs used a Top-K SAE with `K=16`, width multiplier `16`, 300M training tokens, and 10M validation tokens.

<details>
<summary>Commands</summary>

```sh
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux_sub
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_sub --aux-k-coef 0
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_aux --no-subtract-pre-bias
python3 train.py --train-tokens 300000000 --checkpoint-every 150000000 --output-dir artifacts/300M_plain --no-subtract-pre-bias --aux-k-coef 0
```

</details>

| Pre-bias subtraction | AuxK | Validation MSE ↓ | Explained variance ↑ | Dead features ↓ | Active / 10,240 |
|:---:|:---:|---:|---:|---:|---:|
| ✓ | ✓ | **0.002392** | **99.429%** | **0.10%** | **10,230** |
| ✓ | ✗ | 0.002506 | 99.402% | 2.10% | 10,025 |
| ✗ | ✓ | 0.002483 | 99.408% | 14.82% | 8,722 |
| ✗ | ✗ | 0.004031 | 99.038% | 91.88% | 831 |
