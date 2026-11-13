# Knowledge Base
Status: LOCKED
Source: 项目实际代码、F1 SSOT 和用户指令；无新增论文。
方法锚：G1 unitree_rl_lab 的 velocity_env_cfg.py、agents/rsl_rl_ppo_cfg.py 与 mdp/。Isaac Lab manager-based velocity，RSL-RL PPO；actor 只含角速度、重力、指令、关节状态和上一动作历史，线速度真值仅在 critic。
方法原点不含训练副本后加的 arm_swing / shoulder envelope；效果基准是部署 11k，而非当前 Option B。
F1 使用 knee_pitch、lumbar、elbow_yaw 与 wrist_pitch/roll，不能用 G1 相同维数推断语义或顺序。
配置数值逐项来自源码/URDF，平台几何重算。旧 X1/F1 辨识不能当新 F1 实测。上肢 collision 覆盖不足，本机无 Isaac Lab，动态效果未知。
