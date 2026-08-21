from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator


STYLE = {
    "axes.axisbelow": True,
    "axes.edgecolor": "#CCCCCC",
    "axes.facecolor": "#FFFFFF",
    "axes.grid": True,
    "axes.labelcolor": "#64748B",
    "axes.labelsize": 12.0,
    "axes.linewidth": 1.25,
    "axes.titlecolor": "#0F172A",
    "axes.titlesize": 12.0,
    "font.size": 12.0,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 1.0,
    "legend.fontsize": 11.0,
    "xtick.color": "#475569",
    "xtick.labelsize": 11.0,
    "ytick.color": "#475569",
    "ytick.labelsize": 11.0,
}


def _despine(axis, *, left: bool = True, right: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(left)
    axis.spines["right"].set_visible(right)


def save_feature_density_plot(metrics_path: Path, output_path: Path) -> None:
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    records.sort(key=lambda record: record["training_tokens"])
    if not records:
        return

    color_positions = np.linspace(0.0, 1.0, len(records) + 2)[1:-1]
    colors = colormaps["viridis"](color_positions)
    with rc_context(STYLE):
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
        axis.tick_params(which="both", length=0)
        axis.legend(frameon=False, loc="best")
        _despine(axis)

    figure.savefig(
        output_path,
        dpi=160,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.2,
    )


def save_training_plot(
    metrics_path: Path,
    checkpoint_metrics_path: Path,
    output_path: Path,
) -> None:
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    checkpoint_records = [
        json.loads(line)
        for line in checkpoint_metrics_path.read_text().splitlines()
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
    with rc_context(STYLE):
        figure = Figure(figsize=(15.0, 4.4), constrained_layout=True, facecolor="#F8FAFC")
        figure.set_constrained_layout_pads(w_pad=0.12, h_pad=0.12, wspace=0.08)
        FigureCanvasAgg(figure)
        mse_axis, feature_axis, kl_axis = figure.subplots(1, 3)
        auxk_axis = feature_axis.twinx()

        mse = values("mse")
        mse_axis.plot(
            tokens,
            mse,
            color="#7C3AED",
            linewidth=1.5,
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
            auxk_line = auxk_axis.plot(
                tokens[after_dead_window],
                auxk_loss[after_dead_window],
                color="#DB2777",
                linewidth=1.5,
                alpha=0.3,
                label="AuxK NMSE",
            )[0]
        mse_axis.set(ylabel="MSE", yscale="log")
        mse_axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        mse_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        auxk_axis.set_ylabel("AuxK NMSE")
        auxk_axis.tick_params(
            axis="y", which="both", colors=STYLE["ytick.color"], length=0
        )
        auxk_axis.grid(False)
        auxk_axis.set_visible(has_auxk)

        dead_feature_pct = values("dead_feature_pct")
        dead_feature_line = feature_axis.plot(
            tokens,
            dead_feature_pct,
            color="#64748B",
            linewidth=1.5,
            label="Dead features",
        )[0]
        dead_feature_limit = 10 if np.nanmax(dead_feature_pct) <= 10 else 100
        feature_axis.set(
            ylabel="Dead features (%)",
            ylim=(0, dead_feature_limit),
        )
        if auxk_line is not None:
            feature_axis.legend(
                handles=[dead_feature_line, auxk_line],
                frameon=False,
                loc="center right",
            )

        kl_tokens = np.asarray(
            [record["training_tokens"] for record in checkpoint_records],
            dtype=float,
        ) / 1_000_000
        downstream_kl = np.asarray(
            [record["downstream_kl"] for record in checkpoint_records],
            dtype=float,
        )
        kl_axis.plot(
            kl_tokens,
            downstream_kl,
            color="#059669",
            linewidth=1.5,
            marker="o",
            markersize=5,
        )
        kl_axis.set_ylabel("KL(base || SAE)")

        for axis, title in zip(
            (mse_axis, feature_axis, kl_axis),
            (
                "Reconstruction error",
                "Dead features and AuxK" if has_auxk else "Dead features",
                "Next-token KL divergence",
            ),
        ):
            axis.set_title(title, loc="left", pad=14)
            axis.set_xlabel("Training tokens (M)")
            axis.margins(x=0.02)
            axis.grid(axis="x", visible=False)
            axis.tick_params(which="both", length=0)
        _despine(mse_axis)
        _despine(feature_axis)
        _despine(kl_axis)
        _despine(auxk_axis, left=False, right=True)

    figure.savefig(
        output_path, dpi=160, facecolor=figure.get_facecolor(),
        bbox_inches="tight", pad_inches=0.2,
    )
