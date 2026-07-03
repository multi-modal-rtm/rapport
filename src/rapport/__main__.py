import logging
import os

# Must be set before numpy/torch/sklearn are imported (by hydra's deps or
# below) -- BLAS libraries read these once at first use. Without this cap,
# sklearn's LogisticRegression (used every epoch in eval/report.py's
# classification_report, and in the probe-floor regression test) spawns a
# full-width OpenBLAS thread pool per call; on this 60-core box that caused
# 12+ minutes of load-average-108 thread oversubscription for a sub-minute
# job (see docs/DIAGNOSIS.md, Recalibration Step 1). Same fix applied here
# so every training run gets it by default, not just the standalone probe
# script.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from rapport.repro import snapshot_environment
from rapport.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    snapshot_environment(run_dir, cfg)

    log.info("run_name=%s", cfg.run_name)
    log.info(
        "experiment.relational=%s experiment.shift=%s experiment.temporal=%s experiment.lora=%s",
        cfg.experiment.relational,
        cfg.experiment.shift,
        cfg.experiment.temporal,
        cfg.experiment.lora,
    )

    if cfg.experiment.relational or cfg.experiment.shift or cfg.experiment.temporal or cfg.experiment.lora:
        raise NotImplementedError(
            "Only the 'speaker_only' architecture (per-speaker GNN, mean-pooling, frozen "
            "backbones — no relational/shift/temporal components, no LoRA) is implemented "
            "so far. Every other config (full, minus_*, *_lora) is a declarative stub "
            "pending the RAPPORT model implementation."
        )

    from rapport.training.trainer import train

    report = train(cfg, run_dir)
    log.info(
        "test_weighted_f1=%.4f test_macro_f1=%.4f test_accuracy=%.4f best_epoch=%d avg_epoch_time_sec=%.2f",
        report["test_weighted_f1"],
        report["test_macro_f1"],
        report["test_accuracy"],
        report["best_epoch"],
        report["avg_epoch_time_sec"],
    )


if __name__ == "__main__":
    main()
