"""Contact-conditioned lateral foot tilt, independent of Isaac imports.

Current F1 URDF/STL sole: link X forward, Y left, Z up.
Quaternion order is Isaac wxyz. World-Z projection of foot Y is R[2,1].
"""
import torch

LOADED_LATERAL_TILT_FORMULA = (
    "moving * sum(positive_world_fz * contact * (2*(w*x+y*z))^2) / sum(positive_world_fz * contact); "
    "zero when no loaded foot"
)

def contact_lateral_tilt_l2(foot_quat_w: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
    """Mean sin²(lateral tilt) over contacting feet, zero in flight.

    Both sides use the same squared geometric cost. Yaw and pure pitch do not
    contribute. No desired joint target, anatomical side, or heading offset is
    introduced into the actor/controller.
    """
    w, x, y, z = foot_quat_w.unbind(dim=-1)
    lateral_vertical = 2.0 * (w*x + y*z)
    mask = contact.to(dtype=foot_quat_w.dtype)
    return (lateral_vertical.square()*mask).sum(dim=-1)/mask.sum(dim=-1).clamp(min=1.0)


def load_weighted_lateral_tilt_l2(foot_quat_w, contact, foot_force_z, moving):
    """Charge loaded feet proportionally during nonzero velocity commands."""
    w, x, y, z = foot_quat_w.unbind(dim=-1)
    tilt_sq = (2.0 * (w * x + y * z)).square()
    load = torch.where(contact, foot_force_z.clamp_min(0), 0)
    total = load.sum(dim=-1)
    denominator = torch.where(total > 0, total, 1)
    return torch.where(moving, (tilt_sq * load).sum(dim=-1) / denominator, 0)


def _foot_state(env, asset_cfg, sensor_cfg):
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    expected = ["left_ankle_roll", "right_ankle_roll"]
    asset_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    sensor_names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
    if asset_names != expected or sensor_names != expected:
        raise ValueError("Foot tilt reward requires the explicitly resolved left/right foot names")
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    return asset.data.body_quat_w[:, asset_cfg.body_ids, :], contact, sensor


def feet_contact_lateral_tilt(env, asset_cfg, sensor_cfg) -> torch.Tensor:
    quat, contact, _ = _foot_state(env, asset_cfg, sensor_cfg)
    return contact_lateral_tilt_l2(quat, contact)


def feet_loaded_lateral_tilt(env, asset_cfg, sensor_cfg, command_name) -> torch.Tensor:
    quat, contact, sensor = _foot_state(env, asset_cfg, sensor_cfg)
    force_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    moving = (env.command_manager.get_command(command_name) != 0).any(dim=-1)
    return load_weighted_lateral_tilt_l2(quat, contact, force_z, moving)
