# nanointerpret

Minimal training and feature interpretability analysis of LLMs with sparse autoencoders.

## Files

- `train.py`: set up the model and data (streamed or cached residuals), calibrate, train the SAE, and plot metrics.
- `record_activations.py`: record SAE feature activations on the recording split and index them by feature.
- `docs/experiments.md`: experiment results and reproduction commands.
- `docs/overview.md`: brief notes on activation layer, width multiplier, and K.
- `src/data.py`: token cache, residual cache, and context batching.
- `src/experiment.py`: run config, calibration, training loop, evaluation, and downstream KL.
- `src/plot.py`: training and feature-density plots.
- `src/runtime.py`: device selection, model loading, and layer-input capture/patching.
- `src/sae.py`: Top-K SAE (including input normalization and pass-through dims), AuxK loss, and running metrics.

## Code style
- Completely ignore backward-compatibility, do not account for it.
- When removing a component, functionality, abstraction, etc... ensure to cleanup leftovers
- Only add comments when the code is not interpretable by itself