# MechBox

Minimal training and feature interpretability analysis of LLMs with sparse autoencoders.


- `train.py`: build token caches, train/evaluate the SAE, save checkpoints, and plot metrics.
- `docs/hyperparams.md`: brief notes on activation layer, width multiplier, and K.
- `src/data.py`: token-cache creation, validation, and context batching.
- `src/experiment.py`: activation capture, normalization, training, evaluation, metrics, and checkpoints.
- `src/plot.py`: training and feature-density plots.
- `src/runtime.py`: PyTorch device selection.
- `src/sae.py`: Top-K SAE and running metrics.
