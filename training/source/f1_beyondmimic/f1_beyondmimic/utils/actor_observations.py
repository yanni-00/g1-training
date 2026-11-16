"""Proprioceptive F1 actor. Root truth belongs only to critic/reward/evaluation."""
import numpy as np
from f1_beyondmimic.contract import load_contract


def contract():
    count = len(load_contract()["model_joint_names"])
    terms = [("command", 2 * count), ("motion_anchor_ori_b", 6), ("base_ang_vel", 3),
             ("joint_pos", count), ("joint_vel", count), ("actions", count)]
    return {"version": "f1_beyondmimic_actor_proprio_v1", "terms": [n for n, _ in terms],
            "term_dimensions": [d for _, d in terms], "dimension": sum(d for _, d in terms),
            "history_length": 1, "requires_position_estimate": False,
            "requires_linear_velocity_estimate": False}


def audit_native_observations(env):
    base = env.unwrapped
    expected = contract()
    manager = base.observation_manager
    body_count = len(base.cfg.commands.motion.body_names)
    critic_dim = expected["dimension"] + 3 + 3 + body_count * (3 + 6)
    if (list(manager.active_terms["policy"]) != expected["terms"]
            or tuple(manager.group_obs_dim["policy"]) != (expected["dimension"],)
            or tuple(manager.group_obs_dim["critic"]) != (critic_dim,)):
        raise RuntimeError("F1 native observation terms/dimensions differ from contract")
    policy = base.cfg.observations.policy
    if policy.motion_anchor_pos_b is not None or policy.base_lin_vel is not None:
        raise RuntimeError("Actor position/linear-velocity truth leakage")
    return expected


def proprioceptive_observation(q_ref, dq_ref, relative_rotation, omega, q, dq, last_action, default_q):
    """Row-major first two rotation columns, matching native motion_anchor_ori_b."""
    obs = np.concatenate([q_ref, dq_ref, np.asarray(relative_rotation)[:, :2].reshape(-1),
                          omega, np.asarray(q) - default_q, dq, last_action]).astype(np.float32)
    if obs.shape != (contract()["dimension"],) or not np.isfinite(obs).all():
        raise ValueError("Invalid F1 actor input")
    return obs
