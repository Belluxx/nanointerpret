# nanointerpret

Minimal training and feature interpretability analysis of LLMs with sparse autoencoders.

## Files

- `train.py`: set up the model and data (streamed or cached residuals), calibrate, train the SAE, and plot metrics.
- `record_activations.py`: record SAE feature activations on the recording split and index them by token and by feature.
- `interpret.py`: name features with an LLM and check each name: detection on held-out text, generated test texts run through the model, one revision on failure, and steering tokens.
- `explore.py`: serve the local feature explorer over a recorded activation index.
- `explorer/`: the explorer's static web UI (overview, feature pages, token search).
- `docs/experiments.md`: experiment results and reproduction commands.
- `docs/interpretation.md`: the feature interpretation pipeline: how to run it, how it works, and its results.
- `docs/overview.md`: brief notes on activation layer, width multiplier, and K.
- `src/data.py`: token cache, residual cache, and context batching.
- `src/experiment.py`: run config, calibration, training loop, evaluation, and downstream KL.
- `src/features.py`: queries over a recorded activation index, the cached per-feature summary (peak tokens, logit effects), and feature interpretations.
- `src/interpret.py`: example sampling (activation-quantile bins, same-token and neighbor negatives), prompts, the judge client, the model probe (activations on new text, KL-calibrated steering), and scores.
- `src/plot.py`: training and feature-density plots.
- `src/runtime.py`: device selection, model loading, and layer-input capture/patching.
- `src/sae.py`: Top-K SAE (including input normalization, sink tokens, and pass-through dims), AuxK loss, and running metrics.

## Code style
- Completely ignore backward-compatibility, do not account for it.
- When removing a component, functionality, abstraction, etc... ensure to cleanup leftovers
- Only add comments when the code is not interpretable by itself