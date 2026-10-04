# Qwen3 1.7B Base

1. Train the SAE:

```sh
python3 train.py \
  --model-id unsloth/Qwen3-1.7B-Base \
  --activation-layer 14 \
  --width-multiplier 16 \
  --train-tokens 500000000 \
  --checkpoint-every 250000000 \
  --validation-tokens 10000000 \
  --validate-every 100000000 \
  --recording-tokens 100000000 \
  --model-dtype bfloat16
```

> [!NOTE]
> If you are memory poor, add these flags: `--model-batch-size 4 --sae-batch-size 1024` (should use 11GB of RAM).

2. Record feature activations:

```sh
python3 record_activations.py --sae-dir artifacts/qwen3-1.7b-base_l14_w16_k32_500m
```
