"""Modal deployment for the visualizer's intervention endpoint.

Prepare the SAE and model cache, then deploy the endpoint:

    modal run scripts/modal_interventions.py --sae-dir artifacts/<experiment>
    modal deploy scripts/modal_interventions.py

Pass the deployed URL to ``scripts/export_visualizer.py --intervention-url``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal


APP_NAME = "nanointerpret"
GPU = "L4"
MAX_PROMPT_CHARACTERS = 10_000

SAE_DIR = Path("/sae")
HF_CACHE_DIR = Path("/cache")
SAE_FILES = ("config.json", "sae_final.pt")

app = modal.App(APP_NAME)
sae_volume = modal.Volume.from_name(f"{APP_NAME}-sae", create_if_missing=True)
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
    .env(
        {
            "HF_HOME": str(HF_CACHE_DIR),
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_python_source("src")
)


def is_finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_request(request, d_sae: int) -> None:
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(request.prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError(
            f"prompt must be at most {MAX_PROMPT_CHARACTERS:,} characters"
        )
    if type(request.feature_id) is not int or not 0 <= request.feature_id < d_sae:
        raise ValueError(f"feature_id must be between 0 and {d_sae - 1}")
    if request.mode not in ("additive", "clamp"):
        raise ValueError("mode must be additive or clamp")
    if not is_finite_number(request.amount):
        raise ValueError("amount must be finite")
    if type(request.max_new_tokens) is not int or not (
        1 <= request.max_new_tokens <= 256
    ):
        raise ValueError("max_new_tokens must be between 1 and 256")
    if not is_finite_number(request.temperature) or not 0 <= request.temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if not is_finite_number(request.top_p) or not 0.01 <= request.top_p <= 1:
        raise ValueError("top_p must be between 0.01 and 1")
    if type(request.top_k) is not int or not 0 <= request.top_k <= 100:
        raise ValueError("top_k must be between 0 and 100")
    if not is_finite_number(request.repetition_penalty) or not (
        1 <= request.repetition_penalty <= 2
    ):
        raise ValueError("repetition_penalty must be between 1 and 2")


@app.function(
    image=image,
    volumes={str(SAE_DIR): sae_volume, str(HF_CACHE_DIR): cache_volume},
    timeout=30 * 60,
)
def cache_model() -> None:
    from huggingface_hub import snapshot_download

    config = json.loads((SAE_DIR / "config.json").read_text())
    snapshot_download(config["model_id"])
    cache_volume.commit()


@app.cls(
    image=image,
    gpu=GPU,
    volumes={str(SAE_DIR): sae_volume, str(HF_CACHE_DIR): cache_volume},
    max_containers=1,
    scaledown_window=5 * 60,
    startup_timeout=15 * 60,
    timeout=5 * 60,
)
class Interventions:
    @modal.enter()
    def load(self) -> None:
        import torch

        from src.interventions import InterventionGenerator

        missing = [name for name in SAE_FILES if not (SAE_DIR / name).is_file()]
        if missing:
            raise RuntimeError(
                "SAE volume is not prepared; run `modal run "
                "scripts/modal_interventions.py --sae-dir artifacts/<experiment>`"
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
            validate_request(request, self.generator.sae.d_sae)
        except (TypeError, ValueError) as error:
            return JSONResponse(status_code=400, content={"error": str(error)})
        return self.generator.generate_pair(request)


@app.local_entrypoint()
def main(sae_dir: str) -> None:
    local_sae_dir = Path(sae_dir).expanduser()
    missing = [name for name in SAE_FILES if not (local_sae_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {', '.join(missing)} in SAE directory: {local_sae_dir}"
        )

    with sae_volume.batch_upload(force=True) as upload:
        for name in SAE_FILES:
            upload.put_file(local_sae_dir / name, f"/{name}")

    print(f"Uploaded SAE from {local_sae_dir}")
    print("Caching Hugging Face model weights...")
    cache_model.remote()
    print("Ready to deploy with `modal deploy scripts/modal_interventions.py`")
