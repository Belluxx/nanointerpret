# nanointerpret

Minimal training and feature interpretability analysis of LLMs with sparse autoencoders.

## Files

- `train.py`: build token/residual caches, train and evaluate the SAE, save checkpoints, and plot metrics.
- `record_activations.py`: record and save SAE feature activations.
- `docs/experiments.md`: experiment results and reproduction commands.
- `docs/overview.md`: brief notes on activation layer, width multiplier, and K.
- `src/data.py`: token/residual-cache validation and batching.
- `src/experiment.py`: residual-cache capture, calibration, training, evaluation, metrics, and checkpoints.
- `src/misc.py`: generic helper functions/utils.
- `src/plot.py`: training and feature-density plots.
- `src/runtime.py`: device selection, model loading, and layer-input capture.
- `src/sae.py`: Top-K SAE and running metrics.

## Code style
- Completely ignore backward-compatibility, do not account for it.
- When removing a component, functionality, abstraction, etc... ensure to cleanup leftovers
- Only add comments when the code is not interpretable by itself