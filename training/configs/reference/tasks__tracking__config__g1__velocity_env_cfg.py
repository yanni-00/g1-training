"""[OMA] Preserve native observations, PPO, robot control and all constraints."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import RewardsCfg
import whole_body_tracking.tasks.tracking.mdp as mdp
from .flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.mdp.velocity_gait import VelocityGaitCommandCfg


@configclass
class VelocityReferenceRewardsCfg(RewardsCfg):
    # Zero by default: old forward/turn tasks retain their original objective.
    feet_contact_planar_speed=RewTerm(
        func=mdp.feet_contact_planar_speed,weight=0.0,
        params={'asset_cfg':SceneEntityCfg('robot',body_names=['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True),
                'sensor_cfg':SceneEntityCfg('contact_forces',body_names=['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True),
                'force_threshold':10.0})
    # Explicit opt-in only: old tasks keep the original weighted objective.
    motion_pelvis_lin_vel=RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,weight=0.0,
        params={'command_name':'motion','std':0.3,'body_names':['pelvis']})


@configclass
class G1VelocityReferenceEnvCfg(G1FlatEnvCfg):
    rewards:VelocityReferenceRewardsCfg=VelocityReferenceRewardsCfg()
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_contact_planar_speed.params['force_threshold']=float(self.scene.contact_forces.force_threshold)
        old=self.commands.motion
        fields=['asset_name','motion_file','anchor_body_name','body_names','resampling_time_range',
                'pose_range','velocity_range','joint_position_range','adaptive_kernel_size',
                'adaptive_lambda','adaptive_uniform_ratio','adaptive_alpha']
        self.commands.motion=VelocityGaitCommandCfg(**{key:getattr(old,key) for key in fields},debug_vis=False)
