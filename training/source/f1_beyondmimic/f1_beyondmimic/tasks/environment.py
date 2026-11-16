import os
from isaaclab.envs import ManagerBasedRLEnv
from f1_beyondmimic.contract import assert_can_run


class F1Environment(ManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        assert_can_run(os.environ.get("F1_BM_EXECUTION_SCOPE", "training"))
        super().__init__(*args, **kwargs)
