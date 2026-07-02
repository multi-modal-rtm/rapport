import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def snapshot_environment(run_dir: str | Path, cfg: DictConfig | None = None) -> None:
    """Dump the resolved config, git commit hash, and pip freeze output into run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg is not None:
        (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    (run_dir / "git_commit.txt").write_text(_git_commit_hash() + "\n")
    (run_dir / "pip_freeze.txt").write_text(_pip_freeze())


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pip_freeze() -> str:
    # Prefer `uv pip freeze` since uv-managed venvs may not have pip installed.
    for cmd in (
        ["uv", "pip", "freeze", "--python", sys.executable],
        [sys.executable, "-m", "pip", "freeze"],
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return ""
