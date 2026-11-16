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
