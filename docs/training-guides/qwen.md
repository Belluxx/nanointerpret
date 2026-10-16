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

3. Optionally, name the features with a local LLM served by [llama-server](https://github.com/ggml-org/llama.cpp), checking each name ([how](../overview.md#feature-interpretation)). `-cram 0` keeps llama-server's prompt cache from filling RAM, and `--sample` names a subset spread over activation density; leave it out to name every feature. The run resumes where it stopped:

```sh
llama-server -m gemma-4-31B-it-Q8_0.gguf --port 8080 -c 32768 -np 4 -cram 0 --reasoning off
python3 interpret.py --sae-dir artifacts/qwen3-1.7b-base_l14_w16_k32_500m --base-url http://127.0.0.1:8080/v1 --sample 500
```

On an M4 Max, gemma-4-31B-it (~40GB of RAM) takes about 36 s per feature, 5 hours for 500. gemma-4-12B-it decodes ~3x faster and is a bit less precise ([results](../experiments.md#feature-interpretation)).

4. Explore the features at http://127.0.0.1:8000 (the first launch summarizes every feature once, which takes under a minute):

```sh
python3 explore.py --sae-dir artifacts/qwen3-1.7b-base_l14_w16_k32_500m
```
