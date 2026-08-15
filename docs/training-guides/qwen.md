# Qwen3 1.7B Base

1. Train the SAE:

```sh
python3 train.py \
  --model-id unsloth/Qwen3-1.7B-Base \
  --activation-layer 14 \
  --width-multiplier 16 \
  --k 16 \
  --train-tokens 500000000 \
  --checkpoint-every 250000000 \
  --validation-tokens 100000000 \
  --model-dtype bfloat16
```

2. Record feature activations:

```sh
python3 record_activations.py --sae-dir artifacts/qwen3_1.7b_l14_w16_k16_500m
```

3. Name the features with an LLM:

```sh
# Around $6-$16 in API cost for 32K features

python3 interpret_features.py \
  --analysis artifacts/qwen3_1.7b_l14_w16_k16_500m/analysis \
  --base-url https://openrouter.ai/api/v1 \
  --api-key "[API_KEY]" \
  --model openai/gpt-5.6-luna \
  --no-reasoning \
  --concurrent 8
```

> [!TIP]
> If you want you can run it without `--no-reasoning` and it will produce higher quality feature titles but at a **MUCH** higher API cost (up to $640 depending on how long the model thinks)

4. Browse the features and their activation contexts:

```sh
python3 visualize.py --analysis artifacts/qwen3_1.7b_l14_w16_k16_500m/analysis
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).
