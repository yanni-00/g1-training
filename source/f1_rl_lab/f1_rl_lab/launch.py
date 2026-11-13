"""Local scaffold gate; actual execution still needs native OMA preflight."""
import json
from pathlib import Path

from .contract import ROOT, load_contract
from .checkpoint import verify_checkpoint


def require_launch(kind, checkpoint=None, root=ROOT):
    root = Path(root)
    cfg = json.loads((root / "configs/launch.json").read_text(encoding="utf-8"))
    load_contract(root)
    if cfg.get("authorization", {}).get(kind) is not True:
        raise RuntimeError(f"F1 {kind} is disabled: current scope is local scaffolding only")
    if kind == "training" and not cfg.get("oma_exp_ref"):
        raise RuntimeError("Register an actual OMA experiment and run native preflight before enabling training")
    if kind == "evaluation" and checkpoint is None:
        raise RuntimeError("Evaluation requires an explicit F1 checkpoint")
    if checkpoint is not None:
        verify_checkpoint(checkpoint, root)
    return cfg
