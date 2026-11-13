"""Register only the F1 velocity task, after starting Isaac Sim."""
import gymnasium as gym

TASK = "F1-29dof-Velocity"
if TASK not in gym.registry:
    gym.register(
        id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "f1_rl_lab.tasks.locomotion.robots.f1.velocity_env_cfg:RobotEnvCfg",
            "play_env_cfg_entry_point": "f1_rl_lab.tasks.locomotion.robots.f1.velocity_env_cfg:RobotPlayEnvCfg",
            "rsl_rl_cfg_entry_point": "f1_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )
