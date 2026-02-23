from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator



def plot_preprocessing_hist(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), dpi=150)

    rows_raw = summary["rows_raw"].to_numpy(dtype=float)
    rows_used = summary["rows_used"].to_numpy(dtype=float)
    feats_raw = summary["features_raw"].to_numpy(dtype=float)
    feats_final = summary["features_final"].to_numpy(dtype=float)

    row_bins_raw = min(12, max(4, int(np.sqrt(len(rows_raw))) + 2))
    row_bins_used = min(12, max(4, int(np.sqrt(len(rows_used))) + 2))

    feat_min = int(min(np.min(feats_raw), np.min(feats_final)))
    feat_max = int(max(np.max(feats_raw), np.max(feats_final)))
    feat_edges = np.arange(feat_min - 0.5, feat_max + 1.5, 1.0)

    axes[0, 0].hist(rows_raw, bins=row_bins_raw, alpha=0.8, color="#2563eb")
    axes[0, 0].set_title("Rows before preprocessing")
    axes[0, 0].set_xlabel("rows")
    axes[0, 0].set_ylabel("datasets")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].hist(rows_used, bins=row_bins_used, alpha=0.8, color="#16a34a")
    axes[0, 1].set_title("Rows after preprocessing")
    axes[0, 1].set_xlabel("rows")
    axes[0, 1].set_ylabel("datasets")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].hist(feats_raw, bins=feat_edges, alpha=0.8, color="#7c3aed", rwidth=0.9)
    axes[1, 0].set_title("Features before preprocessing")
    axes[1, 0].set_xlabel("features")
    axes[1, 0].set_ylabel("datasets")
    feat_tick_step = max(1, int(np.ceil((feat_max - feat_min + 1) / 8)))
    feature_ticks = np.arange(feat_min, feat_max + 1, feat_tick_step)
    axes[1, 0].set_xticks(feature_ticks)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].hist(feats_final, bins=feat_edges, alpha=0.8, color="#f97316", rwidth=0.9)
    axes[1, 1].set_title("Features after preprocessing")
    axes[1, 1].set_xlabel("features")
    axes[1, 1].set_ylabel("datasets")
    axes[1, 1].set_xticks(feature_ticks)
    axes[1, 1].grid(alpha=0.25)

    for ax in axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.tick_params(axis="x", labelrotation=30)

    fig.suptitle("Step 1: dataset shape before/after preprocessing", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def plot_average_learning_curves(curves: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=150)

    palette = {
        "random_init": "#2563eb",
        "g_supervised_init": "#16a34a",
        "g_dynamic_init": "#dc2626",
    }

    for name, curve in curves.items():
        epochs = np.arange(len(curve))
        color = palette.get(name)
        ax.plot(epochs, curve, linewidth=2.0, label=name, color=color)

    ax.set_title("Step 9: average learning curves (balanced accuracy)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def plot_stage_curves_with_mean(
    curves: list[list[float]],
    out_path: Path,
    title: str,
    mean_label: str,
    mean_color: str,
) -> None:
    matrix = np.asarray(curves, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("expected a non-empty list of equally sized curves")

    epochs = np.arange(matrix.shape[1])
    mean_curve = matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=150)
    for row in matrix:
        ax.plot(epochs, row, color="#64748b", alpha=0.22, linewidth=1.1)

    ax.plot(epochs, mean_curve, color=mean_color, linewidth=2.8, label=mean_label)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic_curve(history: pd.DataFrame, out_path: Path, window: int = 50) -> None:
    value_column = "eval_balanced_accuracy" if "eval_balanced_accuracy" in history.columns else "balanced_accuracy"
    values = history[value_column].to_numpy(dtype=float)
    x = np.arange(1, len(values) + 1)
    if len(values) >= window:
        kernel = np.ones(window, dtype=float) / float(window)
        smooth = np.convolve(values, kernel, mode="valid")
        smooth_x = np.arange(window, len(values) + 1)
    else:
        smooth = values
        smooth_x = x

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    ax.plot(x, values, alpha=0.25, color="#475569", linewidth=1.0, label="raw")
    ax.plot(smooth_x, smooth, color="#0f766e", linewidth=2.2, label=f"moving avg ({window})")
    ax.set_title("Step 7: dynamic hypernetwork learning curve")
    ax.set_xlabel("meta-step")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def plot_final_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)

    labels = summary["stage"].tolist()
    means = summary["mean_final_balanced_accuracy"].to_numpy(dtype=float)
    stds = summary["std_final_balanced_accuracy"].to_numpy(dtype=float)

    colors = ["#2563eb", "#16a34a", "#dc2626"]
    ax.bar(labels, means, yerr=stds, color=colors[: len(labels)], alpha=0.9, capsize=4)
    ax.set_title("Final quality by stage")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
