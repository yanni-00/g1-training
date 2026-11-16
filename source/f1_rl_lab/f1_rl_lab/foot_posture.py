"""Contact-conditioned lateral foot tilt, independent of Isaac imports.

Current F1 URDF/STL sole: link X forward, Y left, Z up.
Quaternion order is Isaac wxyz. World-Z projection of foot Y is R[2,1].
"""
import torch


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


def feet_contact_lateral_tilt(env, asset_cfg, sensor_cfg) -> torch.Tensor:
    """Isaac reward adapter; original bound G1 reward module stays byte exact."""
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    expected = ["left_ankle_roll", "right_ankle_roll"]
    asset_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    sensor_names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
    if asset_names != expected or sensor_names != expected:
        raise ValueError("Foot tilt reward requires the explicitly resolved left/right foot names")
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    return contact_lateral_tilt_l2(asset.data.body_quat_w[:, asset_cfg.body_ids, :], contact)
