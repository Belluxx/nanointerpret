# Gemma 3 270M

1. Train the SAE:

```sh
python3 train.py \
  --model-id unsloth/gemma-3-270m \
  --activation-layer 9 \
  --width-multiplier 16 \
  --train-tokens 500000000 \
  --checkpoint-every 250000000 \
  --validation-tokens 10000000
```

> [!NOTE]
> If you are memory poor, add these flags: `--model-batch-size 4 --sae-batch-size 1024` (should use 3.5GB of RAM).

2. Record feature activations:

```sh
python3 record_activations.py --sae-dir artifacts/gemma-3-270m_l9_w16_k32_500m
```

3. Optionally, name the features with a local LLM and check each name: see [Feature interpretation](../interpretation.md).

4. Explore the features at http://127.0.0.1:8000 (the first launch summarizes every feature once, which takes under a minute):

```sh
python3 explore.py --sae-dir artifacts/gemma-3-270m_l9_w16_k32_500m
```
