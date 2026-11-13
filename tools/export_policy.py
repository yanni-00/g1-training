"""Export an actual bound F1 actor; synthetic validation stays in a temporary directory."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))

from f1_rl_lab.checkpoint import verify_checkpoint
from f1_rl_lab.contract import sha256
from f1_rl_lab.policy import load_actor


def export(checkpoint, output):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    identity = verify_checkpoint(checkpoint)
    runtime = identity["runtime"]
    actor = load_actor(checkpoint, runtime)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = output / "policy.onnx"
    torch.onnx.export(actor, torch.zeros(1, runtime["observation_dim"]), str(model),
                      input_names=["obs"], output_names=["actions"], opset_version=17, dynamo=False)
    proto = onnx.load(str(model))
    onnx.helper.set_model_props(proto, {"format": "f1_fullbody_velocity_v1",
                                       "checkpoint_sha256": identity["checkpoint_sha256"],
                                       "urdf_sha256": runtime["urdf_sha256"]})
    onnx.checker.check_model(proto)
    onnx.save(proto, str(model))
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    # Reuse the project's G1 deployment parity protocol, not a new behavior acceptance gate.
    reference = json.loads((ROOT / "configs/reference/g1_11k_manifest.json").read_text(encoding="utf-8"))
    protocol = reference["export_check"]
    rng = np.random.default_rng(reference["checkpoint_iter"])
    maximum = 0.0
    for _ in range(protocol["input_cases"]):
        x = rng.standard_normal((1, runtime["observation_dim"])).astype(np.float32)
        with torch.inference_mode():
            expected = actor(torch.from_numpy(x)).numpy()
        actual = session.run(None, {"obs": x})[0]
        np.testing.assert_allclose(actual, expected, atol=protocol["atol"], rtol=protocol["rtol"])
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    report = {"checkpoint_sha256": identity["checkpoint_sha256"], "policy_sha256": sha256(model),
              "runtime": runtime, "export_check": {**protocol, "max_abs_action_difference": maximum},
              "scope": "CPU inference parity only; no locomotion validation"}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output)["export_check"]))
