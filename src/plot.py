import json
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

from .experiment import DOWNSTREAM_KL_TOKENS

INK = "#1A1A1A"
MUTED = "#8C8C8C"
TRAIN_COLOR = "#C9C9C9"
STYLE = {
    "axes.edgecolor": "#BDBDBD",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 14.0,
    "axes.titlesize": 11.5,
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.0,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.labelcolor": MUTED,
    "xtick.color": MUTED,
    "xtick.major.size": 3.0,
    "xtick.minor.size": 0.0,
    "ytick.color": MUTED,
    "ytick.major.size": 0.0,
}
TRAIN_LINE = {"color": TRAIN_COLOR, "linewidth": 0.8}
VALIDATION_LINE = {"color": INK, "linewidth": 1.4, "marker": "s", "markersize": 3.5}


def save_plots(output_dir: Path) -> None:
    train = read_jsonl(output_dir / "train_metrics.jsonl")
    evaluations = read_jsonl(output_dir / "evaluation_metrics.jsonl")
    config = json.loads((output_dir / "config.json").read_text())
    if train and evaluations:
        save_training_plot(train, evaluations, config, output_dir / "training_metrics.png")
        save_feature_density_plot(evaluations, output_dir / "validation_feature_density.png")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def column(records: list[dict], field: str) -> np.ndarray:
    # Missing values (None) become NaN, which matplotlib skips.
    return np.asarray([record[field] for record in records], dtype=float)


def format_count(count: float, _=None) -> str:
    count = float(f"{count:.3g}")
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if count >= scale:
            return f"{count / scale:g}{suffix}"
    return f"{count:g}"


def new_figure(figsize: tuple[float, float]) -> Figure:
    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.3)


def save_feature_density_plot(evaluations: list[dict], path: Path) -> None:
    # Earlier checkpoints are gray, lighter when earlier; the final one is ink.
    colors = [str(shade) for shade in np.linspace(0.85, 0.6, len(evaluations) - 1)] + [INK]
    with rc_context(STYLE):
        figure = new_figure((8.0, 4.2))
        axis = figure.subplots()
        for record, color in zip(evaluations, colors):
            log_edges = np.asarray(record["feature_density_log10_bin_edges"])
            # Divide by bin width so the wider lowest bin stays comparable.
            density = 100.0 * np.asarray(record["feature_density_bin_counts"]) / record["total_features"] / np.diff(log_edges)
            final = color == INK
            axis.stairs(
                density, 100.0 * 10.0**log_edges, color=color, linewidth=1.5 if final else 0.9, zorder=3 if final else 2,
                label=f"{format_count(record['training_tokens'])} tokens",
            )

        validation_tokens = format_count(evaluations[-1]["tokens"])
        axis.set_title(f"Feature activation frequency on {validation_tokens} validation tokens, % of features per decade")
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{np.format_float_positional(value, trim='-')}%"))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_ylim(0, None)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles[::-1], labels[::-1], loc="upper right")
        save_figure(figure, path)


def save_training_plot(train: list[dict], evaluations: list[dict], config: dict, path: Path) -> None:
    train_tokens = column(train, "tokens")
    validation_tokens = column(evaluations, "training_tokens")
    train_fvu = 1 - column(train, "explained_variance")
    validation_fvu = 1 - column(evaluations, "explained_variance")
    kl = column(evaluations, "downstream_kl")
    kl_ci = np.asarray([record["downstream_kl_ci"] for record in evaluations])
    with rc_context(STYLE):
        figure = new_figure((14.0, 3.8))
        figure.get_layout_engine().set(w_pad=0.2, wspace=0.12)
        fvu_axis, dead_axis, kl_axis = figure.subplots(1, 3)
        dead_axis.sharex(fvu_axis)

        panels = (
            (fvu_axis, "Unexplained variance", train_fvu, validation_fvu, ".3f"),
            (dead_axis, "Dead features, %", column(train, "dead_feature_pct"), column(evaluations, "dead_feature_pct"), ".1f"),
            (kl_axis, "Downstream KL, nats", None, kl, ".3f"),
        )
        for axis, title, train_values, validation_values, value_format in panels:
            if train_values is not None:
                axis.plot(train_tokens, train_values, **TRAIN_LINE)
            axis.plot(validation_tokens, validation_values, **VALIDATION_LINE)
            axis.annotate(
                f"{validation_values[-1]:{value_format}}", (validation_tokens[-1], validation_values[-1]), xytext=(7, 0),
                textcoords="offset points", color=INK, va="center", annotation_clip=False,
            )
            axis.set_title(title)
            axis.margins(x=0.03)
            axis.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
            axis.xaxis.set_major_formatter(FuncFormatter(format_count))
            axis.yaxis.set_major_locator(MaxNLocator(4))

        # Frame the y-axis on training after the initial transient, which would otherwise flatten the rest.
        settled = np.concatenate([train_fvu[train_tokens >= 0.05 * train_tokens[-1]], validation_fvu])
        low, high = np.nanmin(settled), np.nanmax(settled)
        fvu_axis.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
        dead_axis.set_ylim(0, None)
        kl_band = kl_axis.fill_between(validation_tokens, kl_ci[:, 0], kl_ci[:, 1], color=INK, alpha=0.08, linewidth=0)

        log_interval = np.median(np.diff(train_tokens, prepend=0))
        kl_tokens = DOWNSTREAM_KL_TOKENS // config["context_size"] * config["context_size"]
        figure.legend(
            [fvu_axis.lines[0], fvu_axis.lines[1], kl_band],
            [
                f"Train ({format_count(log_interval)}-token averages, {format_count(config['dead_window'])} dead window)",
                f"Validation ({format_count(evaluations[-1]['tokens'])} held-out tokens)",
                f"95% CI (KL on {format_count(kl_tokens)} tokens)",
            ],
            loc="outside upper left",
            ncols=3,
            columnspacing=2.5,
        )
        save_figure(figure, path)
