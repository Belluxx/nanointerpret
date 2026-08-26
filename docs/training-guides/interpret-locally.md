If you don't want to use remote LLMs, choose a 4-bit model based on your available RAM ([quality experiment](../experiments.md#finding-the-best-model-for-interpreting-features-locally)):

| Model | RAM | Quality |
|---|---:|---|
| [Gemma 4 E2B](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) | ~5GB | Lowest |
| [Gemma 4 E4B](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/main) | ~7GB | Medium |
| [Gemma 4 26B-A4B](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main) | ~18GB | Best |

1. Install [llama.cpp](https://github.com/ggml-org/llama.cpp), download the chosen GGUF, and run it:

```sh
llama-server -m /path/to/model.gguf \
  -ngl 999 -fa on -c 6000 --port 9000 --cache-ram 0 \
  --temp 0.85 --top-k 20 --top-p 0.87 --min-p 0 \
  --repeat-penalty 1 --presence-penalty 0
```

2. In another terminal, run the interpret script against the local llama.cpp server.

```sh
python3 interpret_features.py \
  --activations your_sae_dir/activations \
  --base-url http://127.0.0.1:9000/v1 \
  --api-key llamacpp \
  --model model_default \
  --concurrent 4
```
