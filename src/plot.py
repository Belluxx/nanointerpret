from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


FEATURES = (
    ("Dead", "dead_feature_pct", "#64748B"),
    ("Rare", "window_rare_feature_pct", "#F59E0B"),
    ("Overactive", "window_overactive_feature_pct", "#10B981"),
)
STYLE = {
    "axes.facecolor": "#FFFFFF",
    "axes.labelcolor": "#64748B",
    "axes.titlecolor": "#0F172A",
    "grid.color": "#E2E8F0",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
}


def save_training_plot(metrics_path: Path, output_path: Path) -> None:
    """Save training metrics as a two-panel PNG."""
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        return

    def values(field: str) -> np.ndarray:
        return np.asarray([record[field] for record in records], dtype=float)

    tokens = values("tokens") / 1_000_000
    with sns.axes_style("whitegrid", rc=STYLE), sns.plotting_context("notebook"):
        figure = Figure(figsize=(11, 4.4), constrained_layout=True, facecolor="#F8FAFC")
        figure.set_constrained_layout_pads(w_pad=0.12, h_pad=0.12, wspace=0.08)
        FigureCanvasAgg(figure)
        mse_axis, feature_axis = figure.subplots(1, 2)

        sns.lineplot(
            x=tokens, y=values("mse"), color="#2563EB",
            linewidth=2.4, errorbar=None, ax=mse_axis,
        )
        mse_axis.set(ylabel="MSE · log scale", yscale="log")

        for label, field, color in FEATURES:
            sns.lineplot(
                x=tokens, y=values(field), label=label, color=color,
                linewidth=2.2, errorbar=None, ax=feature_axis,
            )
        feature_axis.set(ylabel="Features (%)", ylim=(0, None))
        feature_axis.legend(
            frameon=False, ncol=3, loc="center right",
            bbox_to_anchor=(1, 1.075), borderaxespad=0,
            handlelength=1.4, columnspacing=1.2,
        )

        for axis, title in zip(
            (mse_axis, feature_axis), ("Reconstruction error", "Feature health")
        ):
            axis.set_title(title, loc="left", pad=14)
            axis.set_xlabel("Training tokens (M)")
            axis.margins(x=0.02)
            axis.grid(axis="x", visible=False)
            axis.tick_params(length=0)
        sns.despine(fig=figure)

    figure.savefig(
        output_path, dpi=160, facecolor=figure.get_facecolor(),
        bbox_inches="tight", pad_inches=0.2,
    )
