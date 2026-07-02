import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from socialarcnet.repro import snapshot_environment
from socialarcnet.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    snapshot_environment(run_dir, cfg)

    log.info("run_name=%s", cfg.run_name)
    log.info(
        "experiment.temporal_attention=%s experiment.lora=%s",
        cfg.experiment.temporal_attention,
        cfg.experiment.lora,
    )

    if cfg.experiment.lora or cfg.experiment.temporal_attention:
        raise NotImplementedError(
            "Only the v1 baseline (frozen backbones, no temporal attention, no LoRA — "
            "i.e. config A) is implemented so far. Configs B/C/D land in a later phase."
        )

    from socialarcnet.training.trainer import train

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
