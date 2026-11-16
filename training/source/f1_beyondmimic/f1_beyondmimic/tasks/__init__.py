"""Register only the isolated F1 task. Import after AppLauncher starts."""
import gymnasium as gym

gym.register(
    id="Tracking-VelocityReference-F1-Proprio-v0",
    entry_point="f1_beyondmimic.tasks.environment:F1Environment",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "f1_beyondmimic.tasks.tracking.config.f1_env_cfg:F1EnvCfg",
        "rsl_rl_cfg_entry_point": "f1_beyondmimic.tasks.tracking.config.ppo_cfg:F1PPORunnerCfg",
    },
)
