"""Bind each actual F1 checkpoint to the asset and execution contract."""
import json
from pathlib import Path

from .contract import ROOT, binding, load_contract, sha256


def sidecar_path(checkpoint):
    return Path(str(checkpoint) + ".f1.json")


def write_sidecar(checkpoint, runtime, root=ROOT):
    checkpoint = Path(checkpoint)
    c = load_contract(root)
    data = {"format": "f1_fullbody_checkpoint_v1", "checkpoint_sha256": sha256(checkpoint),
            "robot": c["robot"], "binding": binding(root), "runtime": runtime}
    sidecar_path(checkpoint).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def verify_checkpoint(checkpoint, root=ROOT):
    checkpoint = Path(checkpoint)
    path = sidecar_path(checkpoint)
    if not path.is_file():
        # GM model delivery may return only .pt. The actual producer therefore embeds
        # the same runtime identity before atomically publishing the checkpoint file.
        try:
            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            embedded = saved.get("f1_contract")
        except Exception:
            embedded = None
        if not embedded:
            raise ValueError("Only this F1 Loop's checkpoint is accepted; missing .f1.json and embedded identity")
        data = {**embedded, "checkpoint_sha256": sha256(checkpoint)}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    c = load_contract(root)
    if data.get("format") != "f1_fullbody_checkpoint_v1" or data.get("robot") != c["robot"]:
        raise ValueError("Wrong F1 checkpoint format or robot")
    if data.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Checkpoint bytes do not match the F1 sidecar")
    if data.get("binding") != binding(root):
        raise ValueError("Checkpoint belongs to a different F1 asset/control configuration")
    rt = data["runtime"]
    if (rt["policy_joint_names"] != c["model_joint_names"] or rt["observation_dim"] != c["observation_dim"]
            or rt.get("binding") != data["binding"]):
        raise ValueError("Checkpoint observation/action contract mismatch")
    return data
