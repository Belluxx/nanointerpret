# Experiments

## Pre-bias subtraction and AuxK are both useful

All runs used a Top-K SAE with `K=16`, width multiplier `16`, 100M training tokens, and 10M validation tokens.

<details>
<summary>Commands</summary>

```sh
#!/bin/sh
.venv/bin/python train.py --train-tokens 100000000 --output-dir artifacts/no_sub_auxk
.venv/bin/python train.py --train-tokens 100000000 --output-dir artifacts/sub_auxk --subtract-pre-bias
.venv/bin/python train.py --train-tokens 100000000 --output-dir artifacts/no_sub_no_auxk --aux-k-coef 0
.venv/bin/python train.py --train-tokens 100000000 --output-dir artifacts/sub_no_auxk --subtract-pre-bias --aux-k-coef 0
```

</details>

| Configuration | Validation MSE ↓ | Explained variance ↑ | Dead features ↓ | Active / 10,240 |
|---|---:|---:|---:|---:|
| **Subtract pre-bias + AuxK** | **0.002716** | **99.337%** | **0.99%** | **10,139** |
| Subtract pre-bias, no AuxK | 0.002791 | 99.319% | 6.07% | 9,618 |
| No subtraction + AuxK | 0.003097 | 99.244% | 21.23% | 8,066 |
| Neither | 0.004622 | 98.872% | 88.17% | 1,211 |
