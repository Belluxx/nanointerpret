If you dont't want to use remote LLMs to interpret the SAE features, do the following:

1. Run Gemma 4 MoE on llamacpp server, ensure to install [llama.cpp](https://github.com/ggml-org/llama.cpp) and download [Gemma GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main) before.
```sh
llama-server -m gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  -ngl 999 -fa on -c 6000 --port 9000 \
  --temp 0.85 --top-k 20 --top-p 0.87 --min-p 0 \
  --repeat-penalty 1 --presence-penalty 0 \
  -spec-type draft-mtp   --spec-draft-n-max 2
```

2. Run the interpret script pointing at the local llamacpp server.
```sh
python interpret_features.py \
  --analysis your_sae_dir/activations \
  --base-url http://127.0.0.1:9000/v1 \
  --api-key llamacpp \
  --model model_default \
  --no-reasoning \
  --concurrent 4
```