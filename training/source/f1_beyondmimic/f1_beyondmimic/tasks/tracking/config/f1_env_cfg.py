"""[OMA] design-f1-beyondmimic-v1: explicit F1 selectors and observable actor."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

from f1_beyondmimic.contract import ROOT, load_json, load_contract, effective_control, prepared_motion
from f1_beyondmimic.robots.f1 import robot_cfg
from f1_beyondmimic.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, RewardsCfg
from f1_beyondmimic.tasks.tracking.mdp.velocity_gait import VelocityGaitCommandCfg
from f1_beyondmimic.tasks.tracking.mdp.f1_actions import F1JointPositionAction
from f1_beyondmimic.tasks.tracking.mdp import feet_contact_planar_speed


@configclass
class F1RewardsCfg(RewardsCfg):
    feet_contact_planar_speed = RewTerm(
        func=feet_contact_planar_speed, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll", "right_ankle_roll"], preserve_order=True),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll", "right_ankle_roll"], preserve_order=True),
                "force_threshold": 10.0},
    )


@configclass
class F1EnvCfg(TrackingEnvCfg):
    oma_proprioceptive_actor: bool = True
    rewards: F1RewardsCfg = F1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        c, controls = load_contract(), effective_control()
        inherited = load_json(ROOT / "configs/reference/stable_runtime_settings.json")
        self.sim.dt, self.decimation = inherited["physics_dt"], inherited["decimation"]
        self.sim.render_interval = self.decimation
        self.scene.robot = robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = {name: v["action_scale"] for name, v in controls.items()}
        self.actions.joint_pos.class_type = F1JointPositionAction
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        old = self.commands.motion
        fields = ["asset_name", "resampling_time_range", "pose_range", "velocity_range",
                  "joint_position_range", "adaptive_kernel_size", "adaptive_lambda",
                  "adaptive_uniform_ratio", "adaptive_alpha"]
        self.commands.motion = VelocityGaitCommandCfg(
            **{key: getattr(old, key) for key in fields},
            motion_file=str(prepared_motion()), anchor_body_name=c["anchor_body_name"],
            body_names=c["tracked_body_names"], debug_vis=False,
            yaw_rate_range=tuple(inherited["command_limits"][2]),
            phase_acceleration=inherited["phase_acceleration"],
        )
        curriculum = inherited["training_command_curriculum"]
        self.commands.motion.speed_scale_range = tuple(curriculum["speed_scale_range"])
        self.commands.motion.straight_probability = curriculum["straight_go_probability"]
        self.commands.motion.go_duration_range = tuple(curriculum["go_duration_range_s"])
        self.commands.motion.stop_duration_range = tuple(curriculum["stop_duration_range_s"])
        self.commands.motion.initial_go_probability = curriculum["initial_go_probability"]
        self.episode_length_s = curriculum["episode_length_s"]
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=c["anchor_body_name"])
        # Remap all G1 end-effector selectors; retain their source protections.
        self.terminations.ee_body_pos.params["body_names"] = c["end_effectors"]
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[r"^(?!left_ankle_roll$)(?!right_ankle_roll$)(?!left_wrist_roll_link$)(?!right_wrist_roll_link$).+$"]
        )
        settings = inherited["reward_parameters"]
        for key in ("weight",):
            setattr(self.rewards.motion_body_lin_vel, key, settings["motion_body_lin_vel"][key])
        self.rewards.motion_body_lin_vel.params["std"] = settings["motion_body_lin_vel"]["std"]
        self.rewards.feet_contact_planar_speed.weight = settings["feet_contact_planar_speed"]["weight"]
        self.rewards.feet_contact_planar_speed.params["force_threshold"] = float(self.scene.contact_forces.force_threshold)
