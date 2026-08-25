from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sae_dirs", type=Path, nargs="+")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=Path("feature_quality.md"))
    args = parser.parse_args()

    table = [
        "| SAE | Semantic features | High-score semantic features | Percentage |",
        "|---|---:|---:|---:|",
    ]
    for sae_dir in args.sae_dirs:
        with (sae_dir / "feature_scores.jsonl").open(encoding="utf-8") as file:
            features = [json.loads(line) for line in file if line.strip()]
        semantic = [
            feature for feature in features if feature["category"] == "semantic"
        ]
        high_score = [
            feature
            for feature in semantic
            if feature["score"] is not None and feature["score"] >= args.threshold
        ]
        percentage = 100 * len(high_score) / len(semantic)
        table.append(
            f"| {sae_dir.name} | {len(semantic):,} | {len(high_score):,} | "
            f"{percentage:.2f}% |"
        )

    args.output.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
