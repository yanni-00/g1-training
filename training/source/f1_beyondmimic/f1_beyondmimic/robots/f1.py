"""F1 assets only. Unknown gains are supplied later through identification."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from f1_beyondmimic.contract import ROOT, effective_control, load_contract


def robot_cfg():
    c, controls = load_contract(), effective_control()
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ROOT / c["urdf"]), fix_base=False,
            merge_fixed_joints=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False,
                linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(c["base_position"]),
            joint_pos=dict(zip(c["model_joint_names"], c["default_joint_pos"])),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={name: ImplicitActuatorCfg(
            joint_names_expr=[name], effort_limit_sim=v["effort_limit"],
            velocity_limit_sim=v["velocity_limit"], stiffness=v["stiffness"],
            # Preserve the URDF passive damping contribution in the implicit drive.
            damping=v["pd_damping"] + v["passive_damping"], armature=v["armature"],
        ) for name, v in controls.items()},
    )
