"""Contract failure-mode checks, independent of an installed Isaac simulator."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))

from f1_rl_lab.checkpoint import verify_checkpoint, write_sidecar
from f1_rl_lab.contract import binding, load_contract, effective_control, sha256
from f1_rl_lab.interface import ActorHistory, joint_permutation, joint_targets


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.c = load_contract()

    def terms(self, value):
        return {s["name"]: np.full(s["dim"], value, dtype=np.float32) for s in self.c["observation_terms"]}

    def test_term_major_oldest_first_and_reset(self):
        history = ActorHistory(self.c)
        h = self.c["history_length"]
        for step in range(h + 1):
            result = history.push(self.terms(step))
        offset = 0
        for spec in self.c["observation_terms"]:
            size = h * spec["dim"]
            expected = np.repeat((np.arange(1, h + 1) * spec["scale"])[:, None], spec["dim"], axis=1)
            np.testing.assert_allclose(result[offset:offset + size].reshape(h, spec["dim"]), expected)
            offset += size
        history.reset()
        result = history.push(self.terms(2))
        self.assertEqual(result.size, self.c["observation_dim"])
        np.testing.assert_allclose(result[:h * 3], 2 * self.c["observation_terms"][0]["scale"])

    def test_same_dimension_g1_is_not_f1(self):
        g1 = json.loads((ROOT / "configs/reference/g1_11k_manifest.json").read_text(encoding="utf-8"))["contract"]["policy_joint_names"]
        self.assertEqual(len(g1), len(self.c["model_joint_names"]))
        with self.assertRaises(ValueError):
            joint_permutation(g1, self.c["model_joint_names"])
        reversed_names = self.c["model_joint_names"][::-1]
        p = joint_permutation(reversed_names, self.c["model_joint_names"])
        self.assertEqual([reversed_names[i] for i in p], self.c["model_joint_names"])

    def test_action_finite_and_default(self):
        np.testing.assert_allclose(joint_targets(np.zeros(self.c["action_dim"]), self.c), self.c["default_joint_pos"])
        with self.assertRaises(ValueError):
            joint_targets(np.full(self.c["action_dim"], np.nan), self.c)

    def test_checkpoint_identity_and_tampering(self):
        with tempfile.TemporaryDirectory(prefix="f1_identity_") as temporary:
            path = Path(temporary) / "weights.pt"
            path.write_bytes(b"synthetic fixture, not a model")
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_checkpoint(path)
            runtime = {"binding": binding(), "policy_joint_names": self.c["model_joint_names"],
                       "observation_dim": self.c["observation_dim"]}
            write_sidecar(path, runtime)
            verify_checkpoint(path)
            path.write_bytes(b"different bytes")
            with self.assertRaisesRegex(ValueError, "bytes"):
                verify_checkpoint(path)

    def test_all_entrypoints_deny_before_simulator_import(self):
        # Exercise an explicitly disabled fixture, independent of the current
        # owner's live launch authorization. Never alter the real launch file.
        with tempfile.TemporaryDirectory(prefix="f1_disabled_fixture_") as temporary:
            root = Path(temporary)
            for folder in ("source", "scripts", "tools", "configs"):
                shutil.copytree(ROOT/folder, root/folder, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(ROOT/"resources/f1", root/"resources/f1")
            (root/"configs/launch.json").write_text(json.dumps({"authorization": {}}), encoding="utf-8")
            for entry in ("scripts/rsl_rl/train.py", "scripts/rsl_rl/_isaac_train.py",
                          "scripts/rsl_rl/play.py", "scripts/rsl_rl/_isaac_play.py", "tools/sim2sim.py"):
                args = [sys.executable, str(root / entry)]
                if "play" in entry:
                    args += ["--checkpoint", "not_a_model.pt"]
                if "sim2sim" in entry:
                    args += ["--checkpoint", "not_a_model.pt", "--xml", "not_a_robot.xml", "--output", "unused.csv"]
                completed = subprocess.run(args, capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("disabled", completed.stderr)
                self.assertNotIn("No module named", completed.stderr)

    def test_identification_partial_source_and_asset_binding(self):
        with tempfile.TemporaryDirectory(prefix="f1_id_fixture_") as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            for name in ("robot_contract.json", "control_profile.json", "parameter_identification.json"):
                shutil.copy2(ROOT / "configs" / name, root / "configs" / name)
            asset = root / self.c["urdf"]
            asset.parent.mkdir(parents=True)
            shutil.copy2(ROOT / self.c["urdf"], asset)
            path = root / "configs/parameter_identification.json"
            identification = json.loads(path.read_text(encoding="utf-8"))
            joint = self.c["model_joint_names"][0]
            identification["joints"][joint]["armature"] = 1.0  # synthetic fixture only
            path.write_text(json.dumps(identification), encoding="utf-8")
            self.assertNotEqual(effective_control(root)[joint]["armature"], 1.0)  # pending is not injected
            evidence = root / "synthetic_test_evidence.txt"
            evidence.write_text("unit-test fixture, not hardware identification", encoding="utf-8")
            identification["status"] = "partial"
            identification["sources"] = [{"path": evidence.name, "sha256": sha256(evidence)}]
            path.write_text(json.dumps(identification), encoding="utf-8")
            self.assertEqual(effective_control(root)[joint]["armature"], 1.0)
            identification["status"] = "validated"
            path.write_text(json.dumps(identification), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                effective_control(root)
            identification["status"] = "partial"
            identification["sources"][0]["sha256"] = "wrong"
            path.write_text(json.dumps(identification), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash"):
                effective_control(root)
            identification["urdf_sha256"] = "different robot"
            path.write_text(json.dumps(identification), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different URDF"):
                effective_control(root)


if __name__ == "__main__":
    unittest.main()
