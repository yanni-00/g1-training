# F1 全身行走平台适配
Status: FINAL for local scaffolding
Goal: F1-FULLBODY-WALK-v1
Provenance: Framing-A translation；用户指定原始 G1 方法与 F1 资产，已授权实现。

| 组件 | 处置/依据 |
|---|---|
| PPO/网络/超参/velocity curriculum | 原始源码保留，独立 f1_rl_lab 包 |
| obs/action/history | 构成与缩放保留；全部可动关节 preserve_order，实际 runtime 核对 |
| 模型/选择器 | F1 SSOT 副本；base_link 取代 torso；lumbar/knee_pitch/elbow_*；model_joint_names 不冒充硬件 SDK |
| 初始姿态与高度 | G1 膝弯曲幅度，F1 FK 求平足；上肢世界镜像符号；真实 STL 求足底与高度 |
| PD/armature | G1 同功能关节作工程起点，有逐项来源，不称为 F1 辨识；速度与力矩采用 F1 URDF |
| 被动动力学 | 原 URDF 保留；implicit drive damping=PD derivative+URDF damping；摩擦不重复注入，实际导入语义待验 |
| reward/DR/termination | 原始结构和值；高度/足离地参考用 F1 sole offset 修正，不新增奖励项 |
| 11k/OptionB/摆臂调整 | 不进入初始奖励，后续独立预注册 |
| 导出/续训 | runtime 顺序与全精度契约，资产/配置/权重哈希侧车，拒绝 G1 或其他 F1 权重 |

训练实现 ../../6_sim-train/F1-29DOF/sim_walking/，任务 F1-29dof-Velocity。launch 默认关闭，本次不训练；辨识接口 asset-bound、null 待注入。
本次验证资产、树、FK/平足/限位/镜像、选择器、原始 reward 差量、编译、CPU/ONNX、history、错误权重拒绝、入口与 OMA。
动态训练/new-checkpoint delivery/load/rollout、sim2sim、速度质量和摆臂方向留待授权与环境，不伪造通过。
后续候选变量：原始/11k gait、上肢回零、肩约束形式、稳定性权重、辨识 profile，未来各自有界对照并冻结保护指标。
