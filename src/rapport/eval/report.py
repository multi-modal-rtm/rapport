"""Test-set evaluation: classification report + confusion matrix, saved as json + png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_and_report(y_true: list[int], y_pred: list[int], labels: list[str], run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    label_ids = list(range(len(labels)))

    report_dict = classification_report(
        y_true, y_pred, labels=label_ids, target_names=labels, output_dict=True, zero_division=0
    )
    (run_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("MELD test confusion matrix")
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return {"classification_report": report_dict}
