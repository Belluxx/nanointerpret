# MechBox

Minimal training and feature interpretability analysis of LLMs with sparse autoencoders.


- `train.py`: build token/residual caches, train/evaluate the SAE without the LLM loaded, save checkpoints, and plot metrics.
- `docs/overview.md`: brief notes on activation layer, width multiplier, and K.
- `src/data.py`: token/residual-cache validation and batching.
- `src/experiment.py`: residual-cache capture, normalization, training, evaluation, metrics, and checkpoints.
- `src/plot.py`: training and feature-density plots.
- `src/runtime.py`: PyTorch device selection.
- `src/sae.py`: Top-K SAE and running metrics.

## Notes
- Avoid over-engineering, choose simplicity over complexity and avoid trying to cover every single edge case. Be precise where it matters, and keep the rest simple.
- Run MPS-enabled scripts with the project interpreter outside the sandbox via approved host execution, e.g. `.venv/bin/python train.py --device mps`.
- Completely ignore backward-compatibility, do not account for it.
