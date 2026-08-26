# Gemma 3 270M

1. Train the SAE:

```sh
python3 train.py \
  --model-id unsloth/gemma-3-270m \
  --activation-layer 9 \
  --width-multiplier 16 \
  --k 16 \
  --train-tokens 500000000 \
  --checkpoint-every 250000000 \
  --validation-tokens 10000000
```

> [!NOTE]
> If you are memory poor, add these flags: `--model-batch-size 4 --sae-batch-size 1024` (should use 3.5GB of RAM).

2. Record feature activations:

```sh
python3 record_activations.py --sae-dir artifacts/gemma_3_270m_l9_w16_k16_500m
```

3. Interpret and categorize the features with an LLM (want to do it locally? Check [here](interpret-locally.md)):

```sh
# Around $6 in API cost for 10K features

python3 interpret_features.py \
  --activations artifacts/gemma_3_270m_l9_w16_k16_500m/activations \
  --base-url https://openrouter.ai/api/v1 \
  --api-key "[API_KEY]" \
  --model openai/gpt-5.6-luna \
  --concurrent 8
```

> [!TIP]
> Add `--reasoning` for potentially higher-quality feature interpretations at a **MUCH** higher API cost (up to $200 depending on how long the model thinks).

4. Browse the features and their activation contexts:

```sh
python3 visualize.py --activations artifacts/gemma_3_270m_l9_w16_k16_500m/activations
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).
