![nanointerpret](assets/banner.svg)

Nanointerpret objective is being a minimal but full-fledged interpretability playground where you can:
- Train your own [SAE](https://www.lesswrong.com/posts/8YnHuN55XJTDwGPMr/a-gentle-introduction-to-sparse-autoencoders) on your own LLM
- Automatically interpret SAE features with a local or remote LLM
- Explore the features via a web GUI
- Forcefully activate features and observe the effect on LLM generations (the fun part)

To try it right now, go to [nanointerpret.pages.dev](https://nanointerpret.pages.dev/).

To try it locally with a pretrained SAE, check [Run the visualizer](#run-the-visualizer) below.

To train your SAE locally, check [Train your SAE](#train-your-sae) below.

Want to know more about the decisions that went into making this project? Check [experiments.md](docs/experiments.md).

![example with tree feature clamping](assets/example_trees.svg)

## Run the visualizer

If you don't want to train the model but just explore the features and perform interventions locally:

1. Prepare the environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download the [pretrained Qwen3 SAE and activations](https://huggingface.co/Belluxx/nanointerpret-qwen3) and start the visualizer:

```sh
hf download Belluxx/nanointerpret-qwen3 --local-dir artifacts/nanointerpret-qwen3
python3 visualize.py --activations artifacts/nanointerpret-qwen3/activations
```

3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Train your SAE

1. Prepare the environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Then proceed to a guide:

- [Gemma3 270M Base](docs/training-guides/gemma.md) (~5GB free RAM required)
- [Qwen3 1.7B Base](docs/training-guides/qwen.md) (~14GB free RAM required)

## Details

The repo takes different ideas from [OpenAI](https://arxiv.org/abs/2406.04093) and [Anthropic](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). It also includes various ablation experiments I did to see what works best ([experiments.md](docs/experiments.md))

Future objectives:
- [ ] Use later layers to avoid heavily syntactic features
- [ ] Use llama.cpp to run LLM and capture acivations
- [ ] Test on 3B-7B models
- [ ] Test 1.7B on later layers compraing feature categories and scores
- [ ] Test on interactive chat tuned models (the repo uses pretrained base Gemma/Qwen)

> [!NOTE]
> This should not be taken as a reference implementation. I made this project just to get started with an hands-on approach and share the results publicly.
