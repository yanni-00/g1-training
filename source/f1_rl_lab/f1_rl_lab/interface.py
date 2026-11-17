"""CPU counterpart of the explicit F1 actor/action contract."""
import numpy as np


def joint_permutation(source_names, destination_names):
    if len(set(source_names)) != len(source_names) or len(set(destination_names)) != len(destination_names):
        raise ValueError("Duplicate joint names")
    if set(source_names) != set(destination_names):
        raise ValueError("Different joint semantics; dimensions alone are insufficient")
    return [source_names.index(n) for n in destination_names]


class ActorHistory:
    """Per-term circular history, flattened oldest first as in Isaac ObservationManager.

    reset() repeats the first observation, matching first-buffer initialization after reset.
    """
    def __init__(self, contract):
        self.contract = contract
        self.buffers = {}

    def reset(self):
        self.buffers.clear()

    def push(self, terms):
        pieces = []
        h = self.contract["history_length"]
        for spec in self.contract["observation_terms"]:
            name = spec["name"]
            value = np.asarray(terms[name], dtype=np.float32).reshape(-1) * spec["scale"]
            if value.shape != (spec["dim"],) or not np.isfinite(value).all():
                raise ValueError(f"Invalid observation: {name}")
            if name not in self.buffers:
                self.buffers[name] = np.repeat(value[None, :], h, axis=0)
            else:
                self.buffers[name] = np.concatenate((self.buffers[name][1:], value[None, :]), axis=0)
            pieces.append(self.buffers[name].reshape(-1))
        return np.concatenate(pieces)


def joint_targets(actions, contract, action_clip=None):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (contract["action_dim"],) or not np.isfinite(actions).all():
        raise ValueError("Invalid F1 action vector")
    targets = np.asarray(contract["default_joint_pos"]) + actions * contract["action_scale"]
    # Legacy checkpoints preserve their original unbounded target mapping. New
    # clipped checkpoints carry the exact postprocessing rule in runtime_v2.
    if action_clip is not None:
        from .action_limits import validate_position_clip
        validate_position_clip(action_clip, contract)
        targets = np.clip(targets, action_clip["lower"], action_clip["upper"])
    return targets
