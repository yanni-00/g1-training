"""Offline RSL-RL actor loading; no simulator import."""
import ast
from pathlib import Path

import torch

from .contract import ROOT


def actor_spec(root=ROOT):
    path = Path(root) / "source/f1_rl_lab/f1_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BasePPORunnerCfg")
    call = next(n.value for n in cfg.body if isinstance(n, ast.Assign) and n.targets[0].id == "policy")
    values = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    if values["activation"] != "elu":
        raise ValueError("Offline exporter currently requires the frozen G1-family ELU architecture")
    return values


def make_actor(observation_dim, action_dim, root=ROOT):
    dims = [observation_dim, *actor_spec(root)["actor_hidden_dims"], action_dim]
    modules = []
    for index, (nin, nout) in enumerate(zip(dims, dims[1:])):
        modules.append(torch.nn.Linear(nin, nout))
        if index < len(dims) - 2:
            modules.append(torch.nn.ELU())
    return torch.nn.Sequential(*modules)


def load_actor(checkpoint, runtime, root=ROOT):
    if runtime.get("empirical_normalization") is not False:
        raise ValueError("A normalized policy needs its actual normalizer; unsupported by this contract")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(k in saved for k in ("obs_norm_state_dict", "critic_obs_norm_state_dict")):
        raise ValueError("Unexpected checkpoint normalizer; revise and validate the offline contract")
    state = saved["model_state_dict"]
    actor_state = {k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")}
    actor = make_actor(runtime["observation_dim"], runtime["action_dim"], root)
    actor.load_state_dict(actor_state, strict=True)
    if not all(torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("Checkpoint contains non-finite actor parameters")
    return actor.eval()
