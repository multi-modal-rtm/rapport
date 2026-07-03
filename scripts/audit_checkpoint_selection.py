"""Selection-effect audit (diagnostics only): parses per-epoch per-class val
F1 directly out of the existing training log files (no re-training, no new
persistence format) for all 3 speaker_only seeds, plots disgust and fear val
F1 across epochs with the early-stopping-selected checkpoint marked, and
reports max-at-any-epoch vs. value-at-selected-checkpoint per run/class. If
max >> selected, the pathology is checkpoint selection on weighted F1 (which
is dominated by neutral/joy, MELD's majority classes), not model capacity.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path("docs")
SEEDS = (42, 1337, 2024)
# fixed categorical order (dataviz skill palette, light mode)
SEED_COLORS = {42: "#2a78d6", 1337: "#1baf7a", 2024: "#eda100"}

EPOCH_RE = re.compile(
    r"\[epoch (\d+)\] loss=[\d.]+ val_weighted_f1=([\d.]+) val_macro_f1=[\d.]+ "
    r"time=[\d.]+s per_class_f1=(\{.*\})"
)
BEST_EPOCH_RE = re.compile(r"best_epoch=(\d+)")


def parse_log(seed: int) -> tuple[list[dict], int]:
    log_path = LOG_DIR / f"train_speaker_only_seed{seed}.log"
    text = log_path.read_text()

    epochs = []
    for m in EPOCH_RE.finditer(text):
        epoch, val_wf1, per_class_str = m.groups()
        per_class = ast.literal_eval(per_class_str)
        epochs.append({"epoch": int(epoch), "val_weighted_f1": float(val_wf1), **per_class})

    best_epoch_match = BEST_EPOCH_RE.search(text)
    assert best_epoch_match, f"no final best_epoch= line found in {log_path}"
    best_epoch = int(best_epoch_match.group(1))

    # cross-check: best_epoch must equal the argmax of val_weighted_f1 (trainer.py's
    # selection rule -- strict '>' means first epoch to reach the max is kept)
    best_by_argmax = max(epochs, key=lambda e: e["val_weighted_f1"])["epoch"]
    computed_best_val = next(e["val_weighted_f1"] for e in epochs if e["epoch"] == best_epoch)
    argmax_val = max(e["val_weighted_f1"] for e in epochs)
    assert abs(computed_best_val - argmax_val) < 1e-9, (
        f"seed {seed}: logged best_epoch={best_epoch} (val_wf1={computed_best_val}) doesn't match "
        f"argmax epoch={best_by_argmax} (val_wf1={argmax_val}) -- parsing or trainer logic mismatch"
    )

    return epochs, best_epoch


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    report_rows = []

    for class_name, ax in zip(("fear", "disgust"), axes):
        for seed in SEEDS:
            epochs, best_epoch = parse_log(seed)
            xs = [e["epoch"] for e in epochs]
            ys = [e[class_name] for e in epochs]
            color = SEED_COLORS[seed]

            ax.plot(xs, ys, color=color, linewidth=2, label=f"seed {seed}")
            selected_val = next(e[class_name] for e in epochs if e["epoch"] == best_epoch)
            ax.scatter([best_epoch], [selected_val], color=color, s=70, zorder=5, edgecolor="white")

            max_val = max(ys)
            max_epoch = xs[ys.index(max_val)]
            report_rows.append(
                {
                    "seed": seed, "class": class_name, "selected_epoch": best_epoch,
                    "f1_at_selected": selected_val, "max_f1_any_epoch": max_val, "max_f1_epoch": max_epoch,
                }
            )

        ax.set_title(f"val {class_name} F1 by epoch (dot = early-stop-selected checkpoint)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val F1" if class_name == "fear" else "")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = Path("docs/checkpoint_selection_disgust_fear.png")
    fig.savefig(out_path, dpi=150)
    print(f"[audit_checkpoint_selection] wrote {out_path}")

    print(f"\n{'seed':<6} {'class':<10} {'selected_epoch':>15} {'f1_at_selected':>16} "
          f"{'max_f1_any_epoch':>18} {'max_f1_epoch':>13}")
    for row in report_rows:
        print(
            f"{row['seed']:<6} {row['class']:<10} {row['selected_epoch']:>15} "
            f"{row['f1_at_selected']:>16.4f} {row['max_f1_any_epoch']:>18.4f} {row['max_f1_epoch']:>13}"
        )


if __name__ == "__main__":
    main()
