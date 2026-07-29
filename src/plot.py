from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator


STYLE = {
    "axes.facecolor": "#FFFFFF",
    "axes.labelcolor": "#64748B",
    "axes.titlecolor": "#0F172A",
    "grid.color": "#E2E8F0",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
}


def save_feature_density_plot(metrics_path: Path, output_path: Path) -> None:
    """Overlay up to three spaced validation feature-density distributions."""
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    records.sort(key=lambda record: record["training_tokens"])
    if len(records) > 3:
        indices = np.rint(np.linspace(0, len(records) - 1, 3)).astype(int)
        records = [records[index] for index in indices]
    if not records:
        return

    colors = sns.color_palette("viridis", n_colors=len(records))
    with sns.axes_style("whitegrid", rc=STYLE), sns.plotting_context("notebook"):
        figure = Figure(figsize=(8.5, 5.2), constrained_layout=True, facecolor="#F8FAFC")
        FigureCanvasAgg(figure)
        axis = figure.subplots()

        for record, color in zip(records, colors):
            edges = np.asarray(record["feature_density_log10_bin_edges"], dtype=float)
            counts = np.asarray(record["feature_density_bin_counts"], dtype=float)
            total_features = int(record["total_features"])
            percentages = 100.0 * counts / total_features
            training_tokens = int(record["training_tokens"])
            dead_percentage = float(record["dead_feature_pct"])
            label = (
                f"{training_tokens / 1_000_000:g}M train "
                f"({dead_percentage:.1f}% dead)"
            )
            axis.stairs(
                percentages,
                100.0 * np.power(10.0, edges),
                label=label,
                color=color,
                linewidth=2.3,
            )

        axis.set_title("Validation feature density", loc="left", pad=14)
        axis.set_xlabel("Feature activation frequency")
        axis.set_ylabel("All SAE features per bin (%)")
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10))
        axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: f"{np.format_float_positional(value, trim='-')}%"
            )
        )
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
    """Save training metrics as a two-panel PNG."""
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
        figure = Figure(figsize=(10.5, 4.4), constrained_layout=True, facecolor="#F8FAFC")
        figure.set_constrained_layout_pads(w_pad=0.12, h_pad=0.12, wspace=0.08)
        FigureCanvasAgg(figure)
        mse_axis, feature_axis = figure.subplots(1, 2)
        auxk_axis = feature_axis.twinx()

        normalized_mse = values("normalized_mse")
        sns.lineplot(
            x=tokens,
            y=normalized_mse,
            color="#7C3AED",
            linewidth=1.5,
            errorbar=None,
            ax=mse_axis,
        )
        auxk_loss = values("auxk_loss", np.nan)
        config_path = metrics_path.with_name("config.json")
        config = json.loads(config_path.read_text())
        dead_window = float(config["dead_window"])
        previous_token_counts = np.concatenate(([0], token_counts[:-1]))
        after_dead_window = previous_token_counts >= dead_window
        has_auxk = np.isfinite(auxk_loss[after_dead_window]).any()
        auxk_line = None
        if has_auxk:
            sns.lineplot(
                x=tokens[after_dead_window],
                y=auxk_loss[after_dead_window],
                color="#DB2777",
                linewidth=1.5,
                errorbar=None,
                ax=auxk_axis,
            )
            auxk_line = auxk_axis.lines[-1]
            auxk_line.set_label("AuxK NMSE")
        mse_axis.set(ylabel="NMSE", yscale="log")
        mse_axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        mse_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        auxk_axis.set_ylabel("AuxK NMSE")
        auxk_axis.tick_params(axis="y", colors=STYLE["ytick.color"], length=0)
        auxk_axis.grid(False)
        auxk_axis.set_visible(has_auxk)

        sns.lineplot(
            x=tokens,
            y=values("dead_feature_pct"),
            color="#64748B",
            linewidth=1.5,
            errorbar=None,
            ax=feature_axis,
        )
        dead_feature_line = feature_axis.lines[-1]
        dead_feature_line.set_label("Dead features")
        feature_axis.set(ylabel="Dead features (%)", ylim=(0, 100))
        legend_lines = [dead_feature_line]
        if auxk_line is not None:
            legend_lines.append(auxk_line)
        feature_axis.legend(handles=legend_lines, frameon=False, loc="center right")

        for axis, title in zip(
            (mse_axis, feature_axis),
            (
                "Reconstruction error",
                "Dead features and AuxK" if has_auxk else "Dead features",
            ),
        ):
            axis.set_title(title, loc="left", pad=14)
            axis.set_xlabel("Training tokens (M)")
            axis.margins(x=0.02)
            axis.grid(axis="x", visible=False)
            axis.tick_params(length=0)
        sns.despine(ax=mse_axis)
        sns.despine(ax=feature_axis)
        sns.despine(ax=auxk_axis, left=True, right=False)

    figure.savefig(
        output_path, dpi=160, facecolor=figure.get_facecolor(),
        bbox_inches="tight", pad_inches=0.2,
    )
