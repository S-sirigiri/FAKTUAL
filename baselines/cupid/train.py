"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import re
from pathlib import Path
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------- helpers for enforcing a hard epoch ceiling across re-runs ----------

def _pref_run_dir(cfg) -> Path:
    """
    Prefer explicit run dirs from the config, then Hydra's runtime dir.
    """
    for key in ("multi_run.run_dir", "hydra.run.dir"):
        v = OmegaConf.select(cfg, key)
        if v:
            return Path(v).resolve()
    try:
        return Path(HydraConfig.get().runtime.output_dir).resolve()
    except Exception:
        return Path(".").resolve()


def _epoch_from_latest_ckpt(ckpt_dir: Path):
    """
    Try to read (0-based) epoch from checkpoints/latest.ckpt.
    Returns int or None if not available.
    """
    latest = ckpt_dir / "latest.ckpt"
    if not latest.exists():
        return None
    try:
        import torch
        obj = torch.load(latest, map_location="cpu")
        # common spots
        for k in ("epoch", "current_epoch", "trainer_epoch", "global_epoch"):
            if isinstance(obj, dict) and k in obj:
                return int(obj[k])
        # sometimes nested
        if isinstance(obj, dict) and isinstance(obj.get("metadata"), dict):
            md = obj["metadata"]
            for k in ("epoch", "current_epoch", "trainer_epoch", "global_epoch"):
                if k in md:
                    return int(md[k])
    except Exception:
        return None
    return None


_epoch_re = re.compile(r"epoch=(\d+)", re.IGNORECASE)

def _epoch_from_filenames(ckpt_dir: Path):
    """
    Fallback: look for files named like 'epoch=0100-*.ckpt' and take the max.
    Returns (0-based) epoch int or None.
    """
    if not ckpt_dir.exists():
        return None
    best = None
    for p in ckpt_dir.glob("*.ckpt"):
        m = _epoch_re.search(p.name)
        if m:
            try:
                ep = int(m.group(1))
                if best is None or ep > best:
                    best = ep
            except Exception:
                pass
    return best


def _read_completed_epochs(ckpt_dir: Path) -> int:
    """
    Returns how many epochs have completed so far.
    Prefer latest.ckpt contents; fall back to filename parsing.
    """
    ep = _epoch_from_latest_ckpt(ckpt_dir)
    if ep is None:
        ep = _epoch_from_filenames(ckpt_dir)
    # If still None, nothing completed yet.
    return (ep + 1) if ep is not None else 0


def _get_total_ceiling(cfg) -> int:
    """
    Choose a total-epoch ceiling:
    - training.max_total_epochs (if present), else
    - training.num_epochs
    """
    ceiling = OmegaConf.select(cfg, "training.max_total_epochs")
    if ceiling is None:
        # some OmegaConf versions require keyword 'default'
        ceiling = OmegaConf.select(cfg, "training.num_epochs", default=0)
    try:
        return int(ceiling or 0)
    except Exception:
        return 0


# -----------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.joinpath('diffusion_policy', 'config'))
)
def main(cfg: OmegaConf):
    # Resolve immediately so all ${now:} resolvers use the same time.
    OmegaConf.resolve(cfg)

    # Determine run/checkpoint locations and the intended total-epoch ceiling.
    run_dir = _pref_run_dir(cfg)
    ckpt_dir = run_dir / "checkpoints"
    done_flag = run_dir / ".training_done"

    total_ceiling = _get_total_ceiling(cfg)  # hard ceiling across all re-runs

    # 0) If we've already stamped completion, do nothing.
    if done_flag.exists():
        print(f"[CUPID] Training already completed for {run_dir}. Nothing to do.")
        return

    # 1) Determine how many epochs are already done.
    completed = _read_completed_epochs(ckpt_dir)

    if total_ceiling > 0:
        # If we've already hit/exceeded the ceiling, stamp and exit.
        if completed >= total_ceiling:
            run_dir.mkdir(parents=True, exist_ok=True)
            done_flag.write_text(
                f"finished at {datetime.now().isoformat()}\n"
                f"completed_epochs={completed}\n"
                f"ceiling={total_ceiling}\n"
                f"source=pre_run_check\n"
            )
            print(f"[CUPID] Detected completed={completed} ≥ ceiling={total_ceiling}. Nothing to do.")
            return

        # Otherwise, trim this invocation to only the remaining epochs.
        remaining = max(0, total_ceiling - completed) + 1
        with open_dict(cfg):
            cfg.training.num_epochs = remaining
            # Strongly encourage resume to avoid starting from epoch 0 state.
            if OmegaConf.select(cfg, "training.resume") is not True:
                cfg.training.resume = True
        print(f"[CUPID] Enforcing ceiling: completed={completed}, ceiling={total_ceiling}. "
              f"Running remaining num_epochs={remaining} in this invocation.")

    # 2) Run the configured workspace.
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    try:
        workspace.run()
    finally:
        # If we had a ceiling, and this invocation ran the remaining epochs,
        # stamp completion so future re-runs are no-ops.
        if total_ceiling > 0:
            new_completed = _read_completed_epochs(ckpt_dir)
            if new_completed >= total_ceiling:
                try:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    done_flag.write_text(
                        f"finished at {datetime.now().isoformat()}\n"
                        f"completed_epochs={new_completed}\n"
                        f"ceiling={total_ceiling}\n"
                        f"source=post_run_check\n"
                    )
                except Exception:
                    # If writing fails, we still consider the run successful;
                    # next run will fall back to checkpoint inspection.
                    pass


if __name__ == "__main__":
    main()
