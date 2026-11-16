"""Run supplied UMR with a Unicode-safe Windows MuJoCo model loader.

Arguments are forwarded to UMR unchanged. No supplied scripts/assets are edited.
"""
import os
from pathlib import Path
import sys
import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
UMR = TRAINING.parents[2]/"f1_UMR"
os.environ.setdefault("MPLCONFIGDIR", str(TRAINING/".runtime/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True
sys.path[:0] = [str(UMR/"scripts"), str(UMR)]

import mujoco
from build_ssot_retarget import native_model
import retarget_smpl_to_humanoid_surface_vector as retarget


class ModelFactory:
    from_xml_path = staticmethod(native_model)
    from_xml_string = staticmethod(mujoco.MjModel.from_xml_string)


class NativeProxy:
    MjModel = ModelFactory

    def __getattr__(self, name):
        return getattr(mujoco, name)


if __name__ == "__main__":
    retarget.mujoco = NativeProxy()
    if "--out" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--out")+1]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        original_preprocess = retarget.common.preprocess_source_ground_for_retarget

        def capture_source_ground(*args, **kwargs):
            result = original_preprocess(*args, **kwargs)
            np.savez_compressed(output.with_suffix(".source_ground_joints.npz"),
                                joints_world=result[1], ground_z=np.array(result[2]),
                                semantics=np.array("Source-human joint positions after source ground alignment; before robot scaling"))
            return result

        retarget.common.preprocess_source_ground_for_retarget = capture_source_ground
    retarget.main()
