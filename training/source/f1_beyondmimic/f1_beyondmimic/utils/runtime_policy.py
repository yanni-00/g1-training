"""F1-only portable normalized actor. Export does not certify robot behavior."""
import re
import torch
from torch import nn
from f1_beyondmimic.contract import FORMAT


class RuntimeActor(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        state = checkpoint["model_state_dict"]
        runtime = checkpoint["infos"]["f1_runtime"]
        if runtime["format"] != FORMAT:
            raise ValueError("Only an F1 BeyondMimic checkpoint is accepted")
        from .actor_observations import contract
        expected = contract()
        if runtime.get("actor_observation_contract") != expected:
            raise ValueError("F1 observable-actor contract mismatch")
        self.observation_dim = expected["dimension"]
        ids = sorted(int(k.split(".")[1]) for k in state if re.fullmatch(r"actor\.\d+\.weight", k))
        layers = []
        for index, number in enumerate(ids):
            w, b = state[f"actor.{number}.weight"], state[f"actor.{number}.bias"]
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data.copy_(w); layer.bias.data.copy_(b)
            layers.append(layer)
            if index + 1 < len(ids): layers.append(nn.ELU())
        if not layers or layers[0].in_features != self.observation_dim or layers[-1].out_features != 29:
            raise ValueError("Expected metadata-matched BeyondMimic actor-to-29")
        self.actor = nn.Sequential(*layers)
        self.normalized = runtime["actor_obs_normalization"]
        self.eps = runtime["normalizer_epsilon"]
        if self.normalized:
            self.register_buffer("mean", state["actor_obs_normalizer._mean"].clone())
            self.register_buffer("std", state["actor_obs_normalizer._std"].clone())
            if self.eps is None or self.eps <= 0 or self.mean.numel() != self.observation_dim or self.std.numel() != self.observation_dim:
                raise ValueError("Invalid model observation normalizer")
        if any(not torch.isfinite(v).all() for v in self.state_dict().values()):
            raise ValueError("Non-finite runtime actor state")
        if self.normalized and (self.std < 0).any(): raise ValueError("Negative normalization standard deviation")

    def forward(self, obs):
        if self.normalized: obs = (obs - self.mean) / (self.std + self.eps)
        return self.actor(obs)


def verify_runtime_checkpoint(policy, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Constructing Linear layers consumes CPU RNG even when weights are loaded.
    with torch.random.fork_rng(devices=[]):
        actor = RuntimeActor(checkpoint).eval()
    generator = torch.Generator(device="cpu").manual_seed(17)
    observations = torch.randn((64,actor.observation_dim), generator=generator)
    if actor.normalized: observations = observations * actor.std + actor.mean
    device = next(policy.parameters()).device
    # Compare float32 exports without TF32 approximation; restore training flags.
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    training_modes = [(module, module.training) for module in policy.modules()]
    try:
        # Empirical normalizers must not update statistics on synthetic probes.
        policy.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            expected = policy.actor(policy.actor_obs_normalizer(observations.to(device))).cpu()
            actual = actor(observations)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        for module, training in training_modes:
            module.training = training
    # Same parity tolerances as the existing BeyondMimic package exporter.
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    return {"samples":64, "max_abs_error":float((actual-expected).abs().max()),
            "actor_obs_normalization":actor.normalized, "normalizer_epsilon":actor.eps}
