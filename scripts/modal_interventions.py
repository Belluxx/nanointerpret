"""Modal deployment for the visualizer's intervention endpoint.

Prepare the SAE and model cache from a public Hugging Face repository, then
deploy the endpoint:

    modal run scripts/modal_interventions.py --sae-repo-id <owner>/<repo>
    modal deploy scripts/modal_interventions.py

Pass the deployed URL to ``scripts/export_visualizer.py --intervention-url``.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "nanointerpret"
GPU = "L4"
MAX_PROMPT_CHARACTERS = 10_000

HF_CACHE_DIR = Path("/cache")
SAE_DIR = HF_CACHE_DIR / "sae"
SAE_FILES = ("config.json", "sae_final.pt")

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(
    f"{APP_NAME}-huggingface", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi[standard]>=0.115",
        "numpy>=1.26",
        "torch>=2.5",
        "tqdm>=4.66",
        "transformers>=4.50",
    )
    .env({"HF_HOME": str(HF_CACHE_DIR), "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("src")
)


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: cache_volume},
    timeout=30 * 60,
)
def prepare(sae_repo_id: str) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    for name in SAE_FILES:
        hf_hub_download(
            sae_repo_id,
            name,
            local_dir=SAE_DIR,
        )
    config = json.loads((SAE_DIR / "config.json").read_text())
    snapshot_download(config["model_id"])
    cache_volume.commit()


@app.cls(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE_DIR: cache_volume},
    max_containers=1,
    scaledown_window=60,
    startup_timeout=15 * 60,
)
class Interventions:
    @modal.enter()
    def load(self) -> None:
        import torch

        from src.interventions import InterventionGenerator

        missing = [name for name in SAE_FILES if not (SAE_DIR / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing {', '.join(missing)}; prepare the SAE cache before deploying"
            )
        self.generator = InterventionGenerator.from_sae_dir(
            SAE_DIR, torch.device("cuda")
        )

    @modal.fastapi_endpoint(method="POST")
    def generate(self, payload: dict):
        from fastapi.responses import JSONResponse

        from src.interventions import InterventionRequest

        try:
            request = InterventionRequest(**payload)
            if (
                not request.prompt.strip()
                or len(request.prompt) > MAX_PROMPT_CHARACTERS
            ):
                raise ValueError("invalid prompt")
            if not 0 <= request.feature_id < self.generator.sae.d_sae:
                raise ValueError("invalid feature_id")
            if request.mode not in ("additive", "clamp"):
                raise ValueError("invalid mode")
            if not 1 <= request.max_new_tokens <= 256:
                raise ValueError("invalid max_new_tokens")
        except (AttributeError, TypeError, ValueError) as error:
            return JSONResponse(status_code=400, content={"error": str(error)})
        return self.generator.generate_pair(request)


@app.local_entrypoint()
def main(sae_repo_id: str) -> None:
    print(f"Caching SAE and model weights from {sae_repo_id}...")
    prepare.remote(sae_repo_id)
    print("Ready to deploy with `modal deploy scripts/modal_interventions.py`")
