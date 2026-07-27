from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator


FEATURES = (
    ("Dead", "dead_feature_pct", "#64748B"),
)
STYLE = {
    "axes.facecolor": "#FFFFFF",
    "axes.labelcolor": "#64748B",
    "axes.titlecolor": "#0F172A",
    "grid.color": "#E2E8F0",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
}


def select_spaced_validation_records(records: list[dict]) -> list[dict]:
    """Select up to three approximately equidistant validations."""
    records = sorted(records, key=lambda record: record["training_tokens"])
    if len(records) <= 3:
        return records

    indices = np.rint(np.linspace(0, len(records) - 1, 3)).astype(int)
    return [records[index] for index in indices]


def save_feature_density_plot(metrics_path: Path, output_path: Path) -> None:
    """Overlay up to three spaced validation feature-density distributions."""
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    selected = select_spaced_validation_records(records)
    if not selected:
        return

    colors = sns.color_palette("viridis", n_colors=len(selected))
    with sns.axes_style("whitegrid", rc=STYLE), sns.plotting_context("notebook"):
        figure = Figure(figsize=(8.5, 5.2), constrained_layout=True, facecolor="#F8FAFC")
        FigureCanvasAgg(figure)
        axis = figure.subplots()

        for record, color in zip(selected, colors):
            edges = np.asarray(record["feature_density_log10_bin_edges"], dtype=float)
            counts = np.asarray(record["feature_density_bin_counts"], dtype=float)
            centers = (edges[:-1] + edges[1:]) / 2
            total_features = int(record["total_features"])
            percentages = 100.0 * counts / total_features
            training_tokens = int(record["training_tokens"])
            dead_percentage = float(record["dead_feature_pct"])
            label = (
                f"{training_tokens / 1_000_000:g}M train "
                f"({dead_percentage:.1f}% dead)"
            )
            axis.step(
                centers,
                percentages,
                where="mid",
                label=label,
                color=color,
                linewidth=2.3,
            )

        axis.set_title("Validation feature density", loc="left", pad=14)
        axis.set_xlabel("log10(feature activation frequency)")
        axis.set_ylabel("All SAE features per bin (%)")
        axis.grid(axis="x", visible=False)
        axis.tick_params(length=0)
        axis.legend(frameon=False, loc="best")
        sns.despine(fig=figure)

    figure.savefig(
        output_path,
        dpi=160,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.2,
    )


def save_training_plot(metrics_path: Path, output_path: Path) -> None:
    """Save training metrics as a three-panel PNG."""
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        return

    def values(field: str, default: float | None = None) -> np.ndarray:
        if default is None:
            return np.asarray([record[field] for record in records], dtype=float)
        return np.asarray([record.get(field, default) for record in records], dtype=float)

    token_counts = values("tokens")
    tokens = token_counts / 1_000_000
    with sns.axes_style("whitegrid", rc=STYLE), sns.plotting_context("notebook"):
        figure = Figure(figsize=(15, 4.4), constrained_layout=True, facecolor="#F8FAFC")
        figure.set_constrained_layout_pads(w_pad=0.12, h_pad=0.12, wspace=0.08)
        FigureCanvasAgg(figure)
        mse_axis, auxk_axis, feature_axis = figure.subplots(1, 3)

        normalized_mse = values("normalized_mse")
        sns.lineplot(
            x=tokens,
            y=normalized_mse,
            color="#7C3AED",
            linewidth=2.4,
            errorbar=None,
            ax=mse_axis,
        )
        auxk_loss = values("auxk_loss", np.nan)
        config_path = metrics_path.with_name("config.json")
        config = json.loads(config_path.read_text())
        dead_window = float(config["dead_window"])
        previous_token_counts = np.concatenate(([0], token_counts[:-1]))
        after_dead_window = previous_token_counts >= dead_window
        if np.isfinite(auxk_loss[after_dead_window]).any():
            sns.lineplot(
                x=tokens[after_dead_window],
                y=auxk_loss[after_dead_window],
                color="#DB2777",
                linewidth=2.0,
                errorbar=None,
                ax=auxk_axis,
            )
        mse_axis.set(ylabel="NMSE", yscale="log")
        mse_axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        mse_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        auxk_axis.set_ylabel("Normalized error")

        for _, field, color in FEATURES:
            sns.lineplot(
                x=tokens, y=values(field), color=color,
                linewidth=2.2, errorbar=None, ax=feature_axis,
            )
        feature_axis.set(ylabel="Features (%)", ylim=(0, 100))

        for axis, title in zip(
            (mse_axis, auxk_axis, feature_axis),
            ("Reconstruction error", "Normalized AuxK", "Dead features"),
        ):
            axis.set_title(title, loc="left", pad=14)
            axis.set_xlabel("Training tokens (M)")
            axis.margins(x=0.02)
            axis.grid(axis="x", visible=False)
            axis.tick_params(length=0)
        if dead_window:
            auxk_axis.set_xlim(left=dead_window / 1_000_000)
        sns.despine(fig=figure)

    figure.savefig(
        output_path, dpi=160, facecolor=figure.get_facecolor(),
        bbox_inches="tight", pad_inches=0.2,
    )
