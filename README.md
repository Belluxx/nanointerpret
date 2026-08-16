![nanointerpret](assets/banner.svg)

This repo offers a minimal self-contained playground where you can play with sparse autoencoders, training many versions of them with different params and see some simple but cool interpretability results.

![example with tree feature clamping](assets/example_trees.svg)

The main objectives are:
- [x] Take different SAE training ideas from [OpenAI](https://arxiv.org/abs/2406.04093) and [Anthropic](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [x] Cache LLM activations on disk for quick ablation experiments (avoid recomputing activations)
- [x] Do some ablation experiments on different SAE training approaches to see what works best (see [experiments.md](docs/experiments.md))
- [x] Support for feature clamping and additive steering

Secondary objectives:
- [ ] Use GGUF / llama.cpp to run LLM and capture acivations
- [ ] Test on 3B-7B models
- [ ] Test on interactive chat tuned models (the repo uses pretrained base Gemma/Qwen)

## How to run

Prepare the environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then proceed to a guide:

- [Gemma3 270M Base](docs/training-guides/gemma.md) (~5GB free RAM required)
- [Qwen3 1.7B Base](docs/training-guides/qwen.md) (~12GB free RAM required)

## Disclaimer

> [!NOTE]
> This should not be taken as a reference implementation. I made this project just to get started with an hands-on approach and share the results publicly.
