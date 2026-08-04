# NanoInterpret

I find mechanistic interpretability of LLMs extremely interesting, so I built this repo to offer a minimal playground where you can play with Sparse Autoencoders, train many versions with different params and see some simple but cool interpretability results.

> [!NOTE]
> I am by no means an expert in interpretability, I just wanted to get started with an hands-on approach and share the results publicly.

The main objectives are:
- [x] Take different SAE training ideas from [OpenAI](https://arxiv.org/abs/2406.04093) and [Anthropic](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [x] Optimize SAE training for Apple Silicon (MPS)
- [x] Cache LLM activations on disk for quick ablation experiments (avoid recomputing activations)
- [x] Do some ablation experiments on different SAE training approaches to see what works best (see [experiments.md](docs/experiments.md))
- [ ] Use GGUF / llama.cpp to capture acivations
- [ ] Check differences for a SAE trained on Gemma3-270M base version and Gemma3-270M instruct (chat version)
- [ ] Train a 1B tokens SAE

## Main experiment reproduction

1. Train the SAE:

```py
python3 train.py --output-dir artifacts/500M --train-tokens 500000000 --checkpoint-every 250000000 --validation-tokens 10000000
```

2. Extract feature activation stats:

```py
python3 record_activations.py --sae-dir artifacts/500M
```

3. Name the features with an LLM:

```py
# Around $2-$5 in API cost for 10K features

python3 interpret_features.py --analysis artifacts/500M/analysis --base-url https://openrouter.ai/api/v1 --api-key [API_KEY] --model openai/gpt-5.6-luna --no-reasoning --concurrent 8
```

> [!TIP]
> If you want you can run it without `--no-reasoning` and it will produce higher quality feature titles but at a **MUCH** higher API cost (up to $200 depending on how long the model thinks)

4. Browse the features and their strongest activation contexts:

```py
python3 visualize.py --analysis artifacts/500M/analysis
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).
