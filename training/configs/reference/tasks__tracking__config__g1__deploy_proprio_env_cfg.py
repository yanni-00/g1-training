"""Only action processing changes: exact deployed target constraints."""
from isaaclab.utils import configclass
from .proprio_velocity_env_cfg import G1VelocityReferenceProprioEnvCfg
from whole_body_tracking.tasks.tracking.mdp.deployment_actions import DeployableJointPositionAction

@configclass
class G1DeployProprioEnvCfg(G1VelocityReferenceProprioEnvCfg):
    oma_deployment_target_limits:bool=True
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.class_type=DeployableJointPositionAction
