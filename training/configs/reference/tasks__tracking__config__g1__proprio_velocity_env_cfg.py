"""[OMA] Owner-approved deployment-observable BeyondMimic actor; unchanged critic/rewards."""
from isaaclab.utils import configclass
from .velocity_env_cfg import G1VelocityReferenceEnvCfg


@configclass
class G1VelocityReferenceProprioEnvCfg(G1VelocityReferenceEnvCfg):
    oma_proprioceptive_actor: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Same exclusions as the upstream G1FlatWoStateEstimationEnvCfg.
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
