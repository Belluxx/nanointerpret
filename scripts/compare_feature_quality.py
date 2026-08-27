from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import colormaps, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plot import STYLE


CATEGORIES = ("token-specific", "lexical", "semantic")
SCORE_GROUPS = (
    (None, "All"),
    ("token-specific", "Token-specific"),
    ("lexical", "Lexical"),
    ("semantic", "Semantic"),
)
DEFAULT_THRESHOLDS = (0.5, 0.9)
CATEGORY_LABELS = ("Token-specific", "Lexical", "Semantic", "Uninterpretable")
CATEGORY_COLORS = ("#E2E8F0", "#F5C98B", "#059669", "#94A3B8")
CATEGORY_HATCHES = (None, None, None, "///")
CATEGORY_TEXT_COLORS = ("#0F172A", "#0F172A", "#FFFFFF", "#FFFFFF")


@dataclass(frozen=True)
class SAEData:
    name: str
    category_counts: np.ndarray
    scores: dict[str | None, np.ndarray]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def load_sae(path: Path) -> SAEData:
    interpretations = load_jsonl(path / "feature_interpretations.jsonl")
    categories = {
        int(feature["feature_id"]): feature["category"]
        for feature in interpretations
    }
    category_counts = np.asarray(
        [
            sum(value == category for value in categories.values())
            for category in (*CATEGORIES, None)
        ]
    )
    score_groups = {category: [] for category, _ in SCORE_GROUPS}
    for feature in load_jsonl(path / "feature_scores.jsonl"):
        if feature["score"] is None:
            continue
        score = float(feature["score"])
        category = categories.get(int(feature["feature_id"]))
        score_groups[None].append(score)
        if category in CATEGORIES:
            score_groups[category].append(score)

    return SAEData(
        name=path.name,
        category_counts=category_counts,
        scores={
            category: np.asarray(scores)
            for category, scores in score_groups.items()
        },
    )


def format_percentage(value: float) -> str:
    text = np.format_float_positional(
        value,
        precision=2,
        unique=False,
        fractional=False,
        trim="-",
    )
    return f"{text}%"


def percentages(counts: np.ndarray) -> np.ndarray:
    return 100 * counts / counts.sum()


def quality_percentages(
    scores: np.ndarray, thresholds: tuple[float, float]
) -> np.ndarray:
    low, high = thresholds
    counts = np.asarray(
        [
            np.count_nonzero(scores < low),
            np.count_nonzero((scores >= low) & (scores <= high)),
            np.count_nonzero(scores > high),
        ]
    )
    return percentages(counts)


def quality_headers(thresholds: tuple[float, float]) -> tuple[str, str, str]:
    low, high = thresholds
    return (
        f"Score < {low:g}",
        f"{low:g} ≤ score ≤ {high:g}",
        f"Score > {high:g}",
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---" + "|---:" * (len(headers) - 1) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def category_table(saes: list[SAEData]) -> str:
    rows = [
        [
            sae.name,
            *(format_percentage(value) for value in percentages(sae.category_counts)),
        ]
        for sae in saes
    ]
    return markdown_table(["SAE", *CATEGORY_LABELS], rows)


def quality_table(
    saes: list[SAEData], category: str | None, thresholds: tuple[float, float]
) -> str:
    rows = [
        [
            sae.name,
            *(
                format_percentage(value)
                for value in quality_percentages(sae.scores[category], thresholds)
            ),
        ]
        for sae in saes
    ]
    return markdown_table(["SAE", *quality_headers(thresholds)], rows)


def save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=160,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.2,
    )


def save_category_plot(saes: list[SAEData], path: Path) -> None:
    names = [sae.name for sae in saes]
    values = np.vstack([percentages(sae.category_counts) for sae in saes])
    with rc_context({**STYLE, "hatch.color": "#64748B"}):
        figure = Figure(
            figsize=(11, max(3.6, 2.0 + 0.55 * len(saes))),
            constrained_layout=True,
            facecolor="#F8FAFC",
        )
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        left = np.zeros(len(saes))
        legend_bars = []
        for index, (label, color, hatch, text_color) in enumerate(
            zip(
                CATEGORY_LABELS,
                CATEGORY_COLORS,
                CATEGORY_HATCHES,
                CATEGORY_TEXT_COLORS,
            )
        ):
            segment = values[:, index]
            bars = axis.barh(
                names,
                segment,
                left=left,
                height=0.62,
                label=label,
                color=color,
                edgecolor="none",
                hatch=hatch,
                linewidth=0,
            )
            legend_bars.append(bars)
            for bar, value in zip(bars, segment):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    format_percentage(value),
                    color=text_color,
                    fontsize=10,
                    alpha=0.72,
                    ha="center",
                    va="center",
                )
            left += segment

        axis.set_title("Feature categories", loc="left", pad=42)
        axis.set_xlabel("All SAE features")
        axis.set_xlim(0, 100)
        axis.invert_yaxis()
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        axis.grid(axis="y", visible=False)
        axis.tick_params(which="both", length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            handles=legend_bars,
            labels=CATEGORY_LABELS,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncols=len(CATEGORY_LABELS),
            borderaxespad=0,
        )
    save_figure(figure, path)


def mean_confidence_interval(scores: np.ndarray) -> tuple[float, float, float]:
    mean = float(scores.mean())
    margin = 1.96 * float(scores.std(ddof=1) / np.sqrt(len(scores)))
    lower = max(0, mean - margin)
    upper = min(1, mean + margin)
    return mean, mean - lower, upper - mean


def save_score_plot(saes: list[SAEData], path: Path) -> None:
    positions = np.arange(len(SCORE_GROUPS))
    bar_width = 0.8 / len(saes)
    color_positions = np.linspace(0, 1, len(saes) + 2)[1:-1]
    colors = colormaps["viridis"](color_positions)
    with rc_context(STYLE):
        figure = Figure(
            figsize=(11, 5.5),
            constrained_layout=True,
            facecolor="#F8FAFC",
        )
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        legend_bars = []
        upper_limits = []
        for index, (sae, color) in enumerate(zip(saes, colors)):
            statistics = [
                mean_confidence_interval(sae.scores[category])
                for category, _ in SCORE_GROUPS
            ]
            means = [mean for mean, _, _ in statistics]
            errors = np.asarray(
                [
                    [lower for _, lower, _ in statistics],
                    [upper for _, _, upper in statistics],
                ]
            )
            upper_limits.extend(np.asarray(means) + errors[1])
            bars = axis.bar(
                positions - 0.4 + bar_width / 2 + index * bar_width,
                means,
                width=bar_width,
                color=color,
                label=sae.name,
                yerr=errors,
                capsize=3,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
            )
            legend_bars.append(bars)

        axis.set_title("Mean feature quality score", loc="left", pad=42)
        axis.set_ylabel("Mean score")
        axis.set_ylim(0, min(1, max(upper_limits) * 1.2))
        axis.set_xticks(positions, [label for _, label in SCORE_GROUPS])
        axis.grid(axis="x", visible=False)
        axis.tick_params(which="both", length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            handles=legend_bars,
            labels=[sae.name for sae in saes],
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncols=min(3, len(saes)),
            borderaxespad=0,
        )
    save_figure(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare feature categories and intervention quality across SAEs."
    )
    parser.add_argument("sae_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=2,
        default=DEFAULT_THRESHOLDS,
        metavar=("LOW", "HIGH"),
        help="Score cutoffs for the three quality strata. Default: 0.5 0.9.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("feature_quality_results"),
        help="Directory for the Markdown report and plots. Default: feature_quality_results.",
    )
    args = parser.parse_args()

    thresholds = tuple(args.thresholds)
    if not 0 <= thresholds[0] < thresholds[1] <= 1:
        parser.error("--thresholds must be strictly increasing values from 0 to 1")

    saes = [load_sae(path) for path in args.sae_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_category_plot(saes, args.output_dir / "feature_categories.png")
    save_score_plot(saes, args.output_dir / "feature_score_bars.png")

    report = [
        "# Feature quality comparison",
        "",
        "Tables report percentages within each SAE and feature group.",
        "",
        "## Feature categories",
        "",
        category_table(saes),
        "",
        "![Feature category distribution](feature_categories.png)",
        "",
        "## Feature score comparison",
        "",
        "Bars show the mean of successfully scored features; whiskers show 95% "
        "confidence intervals for the mean.",
        "",
        "![Feature score comparison](feature_score_bars.png)",
        "",
        "## Quality stratification",
        "",
        "Quality percentages include successfully scored features only.",
    ]
    for category, label in SCORE_GROUPS:
        report.extend(
            [
                "",
                f"### {label} features",
                "",
                quality_table(saes, category, thresholds),
            ]
        )

    report_path = args.output_dir / "feature_quality.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved feature quality results to {args.output_dir}")


if __name__ == "__main__":
    main()
