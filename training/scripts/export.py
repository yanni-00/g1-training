"""Export an actual F1 checkpoint actor and runtime; no simulator or robot calls."""
import argparse
import json
from pathlib import Path
import torch
from _bootstrap import ROOT
from f1_beyondmimic.contract import sha256
from f1_beyondmimic.utils.checkpoint import load_checkpoint
from f1_beyondmimic.utils.runtime_policy import RuntimeActor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    actor = RuntimeActor(checkpoint).eval()
    output = args.output.resolve()
    if output == args.checkpoint.resolve() or args.checkpoint.resolve().is_relative_to(output):
        raise ValueError("Export directory must be separate from checkpoint input")
    # Refuse to replace an accepted package implicitly.
    output.mkdir(parents=True, exist_ok=False)
    sample = torch.zeros(1, actor.observation_dim)
    with torch.inference_mode():
        traced = torch.jit.trace(actor, sample)
        torch.testing.assert_close(traced(sample), actor(sample))
        traced.save(str(output / "actor.pt"))
    torch.save(checkpoint["infos"]["f1_runtime"], output / "runtime.pt")
    manifest = {"format": "f1_beyondmimic_actor_package_v1", "source_checkpoint_sha256": sha256(args.checkpoint),
                "files": {name: sha256(output / name) for name in ("actor.pt", "runtime.pt")},
                "actor_observation_dim": actor.observation_dim, "real_robot_tested": False,
                "hardware_deployment_approved": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
