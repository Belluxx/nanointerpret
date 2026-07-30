# NanoInterpret

I find mechanistic interpretability of LLMs extremely interesting, so I built this repo to offer a minimal playground where you can play with Sparse Autoencoders, train many versions with different params and see some simple but cool interpretability results.

> [!NOTE]
> I am by no means an expert in interpretability, I just wanted to get started with an hands-on approach and share the results publicly.

The main objectives are:
- [x] Take different SAE training ideas from [OpenAI](https://arxiv.org/abs/2406.04093) and [Anthropic](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [x] Optimize SAE training for Apple Silicon (MPS)
- [x] Do some ablation experiments on different SAE training approaches to see what works best (see [experiments.md](docs/experiments.md))
- [ ] Check differences for a SAE trained on Gemma3-270M base version and Gemma3-270M instruct (chat version)
- [ ] Train a 1B tokens SAE
