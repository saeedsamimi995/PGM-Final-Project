from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .pipeline import HistoryBundle


def plot_training_curves(
    histories: HistoryBundle,
    metrics: Iterable[str] = ("loss", "val_loss"),
) -> plt.Figure:
    """Plot the training and validation metrics for the classifier."""
    history = histories.classifier
    figure, axis = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)))
    axis = np.atleast_1d(axis)

    for ax, metric in zip(axis, metrics):
        values = history.history.get(metric)
        if values is None:
            continue
        ax.plot(values, label=metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    figure.tight_layout()
    return figure


def plot_confusion(
    matrix: np.ndarray,
    class_names: Sequence[str],
    title: str,
) -> plt.Figure:
    """Plot a confusion matrix with annotations."""
    figure, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    max_value = matrix.max()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > max_value / 2 else "black",
            )

    figure.tight_layout()
    return figure

