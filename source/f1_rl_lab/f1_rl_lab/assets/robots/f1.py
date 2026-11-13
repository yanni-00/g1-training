"""F1 asset; source URDF limits and passive dynamics are authoritative."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from f1_rl_lab.contract import ROOT, effective_control, load_contract

C = load_contract()
CONTROL = effective_control()


@configclass
class F1ArticulationCfg(ArticulationCfg):
    # This is a model order; the actual future hardware SDK order is unknown.
    model_joint_names: list[str] = C["model_joint_names"]
    soft_joint_pos_limit_factor = 0.9  # original UnitreeArticulationCfg


F1_29DOF_CFG = F1ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ROOT / C["urdf"]), fix_base=False,
        activate_contact_sensors=True, replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=tuple(C["base_position"]),
        joint_pos=dict(zip(C["model_joint_names"], C["default_joint_pos"])), joint_vel={".*": 0.0}),
    actuators={name: ImplicitActuatorCfg(
        joint_names_expr=[name], effort_limit_sim=v["effort_limit"], velocity_limit_sim=v["velocity_limit"],
        stiffness=v["stiffness"], damping=v["total_damping"], armature=v["armature"])
        for name, v in CONTROL.items()},
)
