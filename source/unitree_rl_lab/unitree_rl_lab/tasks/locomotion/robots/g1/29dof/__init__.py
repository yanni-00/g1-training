import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

for task_suffix, env_class, play_class in (
    ("11kBaseline", "Baseline11kEnvCfg", "Baseline11kPlayEnvCfg"),
    ("ArmEnvelope", "ArmEnvelopeEnvCfg", "ArmEnvelopePlayEnvCfg"),
):
    gym.register(
        id=f"Unitree-G1-29dof-Velocity-{task_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.arm_envelope_env_cfg:{env_class}",
            "play_env_cfg_entry_point": f"{__name__}.arm_envelope_env_cfg:{play_class}",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )
