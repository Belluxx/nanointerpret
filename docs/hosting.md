# Hosting the visualizer

The raw activation directory is intended for offline analysis. Export a compact
static site instead of deploying the `.npy` files:

```sh
python3 export_visualizer.py \
  --activations artifacts/qwen3_1.7b_l14_w16_k16_500m/activations \
  --output dist/qwen3-1.7b \
  --intervention-url https://YOUR-WORKSPACE--YOUR-ENDPOINT.modal.run
```

The output directory contains the webpage, a feature summary, and JSON files
containing 32 features each. It can be uploaded directly to any static host.
Preview it locally with:

```sh
python3 -m http.server 8080 --directory dist/qwen3-1.7b
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080).

The exporter keeps exact feature counts, maxima, context counts, and
histograms. It stores 20 representative 64-token contexts per feature. The
original activation directory is not needed by the exported site. Omit
`--intervention-url` to export a read-only browser without the intervention
sandbox.

The Modal endpoint must:

- accept the intervention request as a JSON `POST`;
- return `{"baseline": "...", "intervened": "..."}`;
- allow the static website's origin with CORS, including the `OPTIONS`
  preflight and `Content-Type` header.

Keep the model and SAE checkpoint in the Modal service. Only the static export
belongs in the webpage deployment.
