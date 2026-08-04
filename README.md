# nanointerpret

This repo to offers a minimal self-contained playground where you can play with sparse autoencoders, trainin many versions of them with different params and see some simple but cool interpretability results.

> [!NOTE]
> I am by no means an expert in interpretability, I just wanted to get started with an hands-on approach and share the results publicly.

The main objectives are:
- [x] Take different SAE training ideas from [OpenAI](https://arxiv.org/abs/2406.04093) and [Anthropic](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [x] Optimize SAE training for Apple Silicon (MPS)
- [x] Cache LLM activations on disk for quick ablation experiments (avoid recomputing activations)
- [x] Do some ablation experiments on different SAE training approaches to see what works best (see [experiments.md](docs/experiments.md))
- [ ] Use GGUF / llama.cpp to capture acivations
- [ ] Support for ablation / other interventions
- [ ] Test not just on base but also on Gemma3-270M instruct (chat version, useful to steer behaviour in interactive mode)

## Main experiment reproduction

1. Train the SAE:

```sh
python3 train.py --output-dir artifacts/500M --train-tokens 500000000 --checkpoint-every 250000000 --validation-tokens 10000000
```

2. Extract feature activation stats:

```sh
python3 record_activations.py --sae-dir artifacts/500M
```

3. Name the features with an LLM:

```sh
# Around $2-$5 in API cost for 10K features

python3 interpret_features.py --analysis artifacts/500M/analysis --base-url https://openrouter.ai/api/v1 --api-key [API_KEY] --model openai/gpt-5.6-luna --no-reasoning --concurrent 8
```

> [!TIP]
> If you want you can run it without `--no-reasoning` and it will produce higher quality feature titles but at a **MUCH** higher API cost (up to $200 depending on how long the model thinks)

4. Browse the features and their strongest activation contexts:

```sh
python3 visualize.py --analysis artifacts/500M/analysis
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).
