"""An explicit 11k control and a bounded shoulder-pitch ablation.

Reference: training configuration at commit 5421b4f. The current velocity task
uses Option B stability weights; these tasks restore the 11k weights. The
treatment changes only the shoulder-pitch posture cost, keeping the 11k arm
tracking target, observations, actions, gait, and PPO configuration.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RewardsCfg, RobotEnvCfg, RobotPlayEnvCfg


@configclass
class Baseline11kRewardsCfg(RewardsCfg):
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )


@configclass
class ArmEnvelopeRewardsCfg(Baseline11kRewardsCfg):
    # Retain the original posture penalty on every other upper-body joint.
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    # Reuse the 11k target amplitude as the envelope, without enlarging it.
    shoulder_pitch_envelope = RewTerm(
        func=mdp.joint_deviation_l1_envelope,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"]),
            "moving_tolerance": 0.08,
            "command_name": "base_velocity",
        },
    )


@configclass
class Baseline11kEnvCfg(RobotEnvCfg):
    rewards: Baseline11kRewardsCfg = Baseline11kRewardsCfg()


@configclass
class Baseline11kPlayEnvCfg(RobotPlayEnvCfg):
    rewards: Baseline11kRewardsCfg = Baseline11kRewardsCfg()


@configclass
class ArmEnvelopeEnvCfg(RobotEnvCfg):
    rewards: ArmEnvelopeRewardsCfg = ArmEnvelopeRewardsCfg()


@configclass
class ArmEnvelopePlayEnvCfg(RobotPlayEnvCfg):
    rewards: ArmEnvelopeRewardsCfg = ArmEnvelopeRewardsCfg()
