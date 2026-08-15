# Gemma 3 270M

> [!WARNING]
> Gemma 3 models are gated on Hugging Face. Before running the commands below you need to accept Google's usage license on the [model page](https://huggingface.co/google/gemma-3-270m). Then run `hf auth login` to login.

1. Train the SAE:

```sh
python3 train.py \
  --model-id google/gemma-3-270m \
  --activation-layer 9 \
  --width-multiplier 16 \
  --k 16 \
  --train-tokens 500000000 \
  --checkpoint-every 250000000 \
  --validation-tokens 10000000
```

2. Extract feature activation stats:

```sh
python3 record_activations.py --sae-dir artifacts/gemma_3_270m_l9_w16_k16_500m
```

3. Name the features with an LLM:

```sh
# Around $2-$5 in API cost for 10K features

python3 interpret_features.py --analysis artifacts/gemma_3_270m_l9_w16_k16_500m/analysis --base-url https://openrouter.ai/api/v1 --api-key [API_KEY] --model openai/gpt-5.6-luna --no-reasoning --concurrent 8
```

> [!TIP]
> If you want you can run it without `--no-reasoning` and it will produce higher quality feature titles but at a **MUCH** higher API cost (up to $200 depending on how long the model thinks)

4. Browse the features and their activation contexts:

```sh
python3 visualize.py --analysis artifacts/gemma_3_270m_l9_w16_k16_500m/analysis
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).
