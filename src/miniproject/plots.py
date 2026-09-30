"""Figures for the report: one fault run of each statistic, with its threshold."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from miniproject.config import FAULT_DESCRIPTION, FIGURES_DIR


def plot_fault_run(
    pca_scores: pd.DataFrame,
    ridge_scores: pd.DataFrame,
    thresholds: dict[str, float],
    fault: int,
    run: int = 1,
    path: Path | None = None,
) -> Path:
    """Plot T2, SPE, and the ridge score on one faulty run.

    The vertical line is sample 20, the last normal sample. The horizontal
    line is that statistic's validation threshold. An alarm also requires the
    two previous samples to clear the line; the figure shows the raw statistic
    so the threshold itself is visible.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    destination = path or (FIGURES_DIR / f"fault{fault}_run{run}_statistics.png")
    series = {
        "T2": (pca_scores, "T2", thresholds["T2"]),
        "SPE": (pca_scores, "SPE", thresholds["SPE"]),
        "ridge score": (ridge_scores, "score", thresholds["ridge"]),
    }
    # Short panels so Figure 1 fits on one report page with its caption.
    figure, axes = plt.subplots(3, 1, figsize=(7.4, 4.8), sharex=True)
    for axis, (title, (frame, column, threshold)) in zip(axes, series.items()):
        part = frame[(frame["faultNumber"] == fault) & (frame["simulationRun"] == run)]
        part = part.sort_values("sample")
        if part.empty:
            raise ValueError(f"No rows for fault {fault} run {run} ({title}).")
        axis.plot(part["sample"], part[column], color="#1f4e79", linewidth=1.0)
        axis.axhline(threshold, color="#b85c38", linewidth=1.0, label="99th percentile")
        axis.axvline(20, color="#5b5b5b", linewidth=0.8, linestyle="--", label="sample 20")
        axis.set_ylabel(title)
        axis.set_xlim(1, 500)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("sample (3 minutes each)")
    description = FAULT_DESCRIPTION.get(fault, "")
    figure.suptitle(f"Fault {fault}, run {run}: {description}", fontsize=11)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    print(f"figure  {destination.name}")
    return destination
