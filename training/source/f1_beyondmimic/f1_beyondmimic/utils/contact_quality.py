"""[OMA] Contact-current body-speed proxy; not a sole contact-point measurement."""
import torch


def contact_planar_speed_cost(force_w, body_velocity_w, threshold):
    """Sum foot-body planar speed only while its current measured force exceeds threshold.

    A foot rotating about a stationary toe can incur this proxy cost. Physical
    contact-point slip and full gait therefore remain separate acceptance checks.
    """
    if force_w.shape != body_velocity_w.shape or force_w.shape[-1] != 3:
        raise ValueError('Contact/velocity body order and shapes must match')
    touching=force_w.norm(dim=-1)>threshold
    return (body_velocity_w[...,:2].norm(dim=-1)*touching).sum(dim=-1)
