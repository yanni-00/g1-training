"""F1 model hard-limit adapter. Actual hardware protection mapping remains pending."""
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class F1JointPositionAction(JointPositionAction):
    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Non-finite F1 policy action")
        super().process_actions(actions)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._processed_actions.copy_(torch.clamp(self._processed_actions, limits[..., 0], limits[..., 1]))
