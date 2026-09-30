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


def save_plots(output_dir: Path) -> None:
    train = read_jsonl(output_dir / "train_metrics.jsonl")
    evaluations = read_jsonl(output_dir / "evaluation_metrics.jsonl")
    if train and evaluations:
        save_training_plot(train, evaluations, output_dir / "training_metrics.png")
        save_feature_density_plot(evaluations, output_dir / "validation_feature_density.png")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def column(records: list[dict], field: str) -> np.ndarray:
    # Missing values (None) become NaN, which matplotlib skips.
    return np.asarray([record[field] for record in records], dtype=float)


def new_figure(figsize: tuple[float, float]) -> Figure:
    figure = Figure(figsize=figsize, constrained_layout=True, facecolor="#F8FAFC")
    FigureCanvasAgg(figure)
    return figure


def style_axis(axis, title: str, xlabel: str) -> None:
    axis.set_title(title, loc="left", pad=14)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", visible=False)
    axis.tick_params(which="both", length=0)
    despine(axis)


def despine(axis, *, left: bool = True, right: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(left)
    axis.spines["right"].set_visible(right)


def save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=160, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.2)


def save_feature_density_plot(evaluations: list[dict], path: Path) -> None:
    colors = colormaps["viridis"](np.linspace(0.0, 1.0, len(evaluations) + 2)[1:-1])
    with rc_context(STYLE):
        figure = new_figure((8.5, 5.2))
        axis = figure.subplots()
        for record, color in zip(evaluations, colors):
            percentages = 100.0 * np.asarray(record["feature_density_bin_counts"]) / record["total_features"]
            edges = 100.0 * np.power(10.0, record["feature_density_log10_bin_edges"])
            label = f"{record['training_tokens'] / 1_000_000:g}M train ({record['dead_feature_pct']:.1f}% dead)"
            axis.stairs(percentages, edges, label=label, color=color, linewidth=2.3)

        style_axis(axis, "Validation feature density", "Feature activation frequency")
        axis.set_ylabel("All SAE features per bin (%)")
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{np.format_float_positional(value, trim='-')}%"))
        axis.legend(frameon=False, loc="best")
    save_figure(figure, path)


def save_training_plot(train: list[dict], evaluations: list[dict], path: Path) -> None:
    tokens = column(train, "tokens") / 1_000_000
    auxk_loss = column(train, "auxk_loss")
    dead_feature_pct = column(train, "dead_feature_pct")
    has_auxk = bool(np.isfinite(auxk_loss).any())
    with rc_context(STYLE):
        figure = new_figure((15.0, 4.4))
        figure.set_constrained_layout_pads(w_pad=0.12, h_pad=0.12, wspace=0.08)
        mse_axis, feature_axis, kl_axis = figure.subplots(1, 3)
        auxk_axis = feature_axis.twinx()

        mse_axis.plot(tokens, column(train, "mse"), color="#7C3AED", linewidth=1.5)
        mse_axis.set(ylabel="MSE", yscale="log")
        mse_axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        mse_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

        dead_line = feature_axis.plot(tokens, dead_feature_pct, color="#64748B", linewidth=1.5, label="Dead features")[0]
        feature_axis.set(ylabel="Dead features (%)", ylim=(0, 10 if np.nanmax(dead_feature_pct, initial=0) <= 10 else 100))
        auxk_axis.set_visible(has_auxk)
        if has_auxk:
            auxk_line = auxk_axis.plot(tokens, auxk_loss, color="#DB2777", linewidth=1.5, alpha=0.3, label="AuxK NMSE")[0]
            auxk_axis.set_ylabel("AuxK NMSE")
            auxk_axis.tick_params(axis="y", which="both", colors=STYLE["ytick.color"], length=0)
            auxk_axis.grid(False)
            feature_axis.legend(handles=[dead_line, auxk_line], frameon=False, loc="center right")

        kl_tokens = column(evaluations, "training_tokens") / 1_000_000
        kl_ci = np.asarray([record["downstream_kl_ci"] for record in evaluations])
        kl_axis.plot(
            kl_tokens, column(evaluations, "downstream_kl"), color="#059669", linewidth=1.5, marker="o", markersize=5, label="Mean KL"
        )
        kl_axis.fill_between(
            kl_tokens, kl_ci[:, 0], kl_ci[:, 1], color="#059669", alpha=0.15, linewidth=0, label="95% context CI"
        )
        kl_axis.set_ylabel("KL(base || SAE)")
        kl_axis.legend(frameon=False, loc="best")

        titles = ("Reconstruction error", "Dead features and AuxK" if has_auxk else "Dead features", "Next-token KL divergence")
        for axis, title in zip((mse_axis, feature_axis, kl_axis), titles):
            style_axis(axis, title, "Training tokens (M)")
            axis.margins(x=0.02)
        despine(auxk_axis, left=False, right=True)
    save_figure(figure, path)
