# nanointerpret

Minimal training and feature interpretability analysis of LLMs with sparse autoencoders.

## Files

- `train.py`: build token/residual caches, train and evaluate the SAE, save checkpoints, and plot metrics.
- `record_activations.py`: record and save SAE feature activations.
- `interpret_features.py`: ask an LLM to give concise titles to SAE features.
- `visualize.py` and `visualizer/`: serve the local feature browser and its frontend.
- `docs/experiments.md`: experiment results and reproduction commands.
- `docs/overview.md`: brief notes on activation layer, width multiplier, and K.
- `src/data.py`: token/residual-cache validation and batching.
- `src/experiment.py`: residual-cache capture, normalization, training, evaluation, metrics, and checkpoints.
- `src/plot.py`: training and feature-density plots.
- `src/runtime.py`: PyTorch device selection.
- `src/sae.py`: Top-K SAE and running metrics.

## Running scripts
Run MPS-enabled scripts with the project interpreter outside the sandbox via approved host execution, e.g. `.venv/bin/python train.py --device mps`.

## Code style
VERY IMPORTANT: Avoid over-engineering, choose simplicity over complexity and avoid trying to cover every single edge case. Be precise where it matters, and keep the rest simple.

Completely ignore backward-compatibility, do not account for it.
