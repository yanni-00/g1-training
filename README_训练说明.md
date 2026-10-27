# unitree_rl_lab_train — G1 训练代码独立副本

> 从 `../unitree_rl_lab`（保持原样不动）剥离出的**训练侧代码副本**，目标：跑通 G1 velocity 训练全路径。
> 原仓库只保留对照，所有训练实验在本副本内进行。

## 目录结构（已裁剪，仅 G1-29dof，共 246 个文件）

```
unitree_rl_lab_train/
├── source/unitree_rl_lab/        # 可安装训练包（pip install -e 用它）
│   └── unitree_rl_lab/
│       ├── tasks/locomotion/     # 仅 velocity 任务；robots/ 仅 g1/29dof
│       ├── assets/robots/        # unitree.py 含其他机器人死定义（无害，见下）
│       └── utils/                # train/play 依赖，保留
├── scripts/
│   ├── rsl_rl/train.py           # ★ 训练入口
│   ├── rsl_rl/play.py            # 回放
│   └── list_envs.py              # 任务注册器（已改：去掉 mimic.robots 导入）
├── resources/unitree_ros/
│   └── robots/g1_description/    # 仅 G1 模型：g1_29dof* 各形态 URDF/MJCF + meshes/
├── unitree_rl_lab.sh             # 官方安装/启动脚本
└── pyproject.toml / README.md / LICENCE
```

已裁剪（相对原仓库）：`tasks/mimic/`、`robots/go2/`、`robots/h1/`、`scripts/mimic/`、`docker/`、`docs/`、`ui_extension_example.py`、unitree_ros 的 `.git` 及全部非 G1 机器人模型、`g1_description` 内的 23dof 文件。

**已知无害残留**：`assets/robots/unitree.py` 中 go2/h1/mimic 的配置类定义仍在（模块级构造、仅字符串路径、无文件访问，不被任何任务引用）——保留它们是为避免对 700 行文件做手术引入风险。

部署侧（deploy/）**不含**在本副本内——上真机用原仓库 `unitree_rl_lab/deploy/`。

## 环境前提（Linux 训练机）

- Linux + RTX GPU（建议与 X1 训练同机或同配置）
- Isaac Sim **5.1.0** + Isaac Lab **2.3.0**（安装指南：isaac-sim.github.io/IsaacLab）
- conda 环境（官方脚本假定 `env_isaaclab`）

## 模型文件（已就位 ✅）

G1 模型已从官方仓库 `unitreerobotics/unitree_ros` clone 到 `resources/unitree_ros/`（2026-10-13，depth 1）。

已核验：
- 训练指向的 URDF 存在：`resources/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`（39 个 joint 标签，29 个驱动关节）
- mesh 引用为相对路径 `meshes/*.STL`，`g1_description/meshes/` 目录齐全（pelvis、d455、背包等 STL 均在）
- 同目录下还备有其他形态 URDF（23dof、lock_waist、mode_11~16、with_dex 等），换任务形态时可直接替换路径

## 安装与训练

```bash
conda activate env_isaaclab
cd unitree_rl_lab_train

# 安装本包（二选一）
./unitree_rl_lab.sh -i                      # 官方方式（含 conda activate.d 配置）
pip install -e source/unitree_rl_lab        # 直接方式

# 训练（G1 29dof velocity 任务）
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity

# 训练后回放查看
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
```

训练输出 log 在 `logs/`（包内 tasks 注册名均含 "Unitree"）。

## 相对原仓库的改动（3 处）

| 文件 | 改动 |
|------|------|
| `source/.../assets/robots/unitree.py:20-27` | `UNITREE_ROS_DIR` 由占位字符串改为**自动解析**到本目录 `resources/unitree_ros`（可用环境变量覆盖）；`UNITREE_MODEL_DIR` 保持占位 |
| 同文件 `UNITREE_G1_29DOF_CFG` | spawn 从 USD 切换为 **URDF**（README 推荐方式，Isaac Sim ≥5.0 支持） |
| `scripts/list_envs.py:61` | 任务导入列表去掉 `mimic.robots`（该目录已裁剪，不去掉会 ModuleNotFoundError） |

其余 58 个文件与原仓库逐字节一致（已 diff 校验）。mimic 任务仍指向 USD 占位路径——跑 velocity 不受影响。

## 后续路线（对应 G1 逆向计划）

1. 跑通：官方 config 原样训练 → 验证训练环境（本轮）
2. 对照：结果与官方预训练 `unitree_rl_lab/deploy/.../velocity/v0/policy.onnx` 真机表现对比
3. 注入：按 Phase 0 实测行为规格替换 reward 参数（`feet_gait` period/offset、`foot_clearance` target、`base_height` target、命令范围）

## 基于 11k 的摆臂消融入口

以下入口补充上面的初始安装说明。原任务 `Unitree-G1-29dof-Velocity` 仍使用当前 Option B 配置。

| 任务 | 配置 |
|------|------|
| `Unitree-G1-29dof-Velocity-11kBaseline` | 按提交 `5421b4f` 重建 11k 对照：`flat_orientation_l2=-5.0`、`feet_slide=-0.2`，肩 pitch 保留原 L1 |
| `Unitree-G1-29dof-Velocity-ArmEnvelope` | 同一对照配置，仅将肩 pitch 的回零惩罚改为有界容差：运动时默认位周围 `±0.08 rad` 内免罚，超出后恢复 L1 斜率；站立时容差为零 |

`0.08 rad` 来自 11k 的摆臂目标幅值；摆臂奖励权重 `0.3`、周期 `1.0 s`、`std=0.15` 均沿用 11k。其他上肢关节仍受原 L1 约束。容差是软惩罚，不能保证实际关节角不越界。

两个任务的观测、动作、网络与原 checkpoint 兼容。均从 `checkpoints/model_10999_factorygait.pt` 启动；下面为本轮两组各设置 `2000` 个追加训练迭代，沿用 GM 任务 `TASK_20261021_040` 的 `taskCodeInfo.startScript` 中的 `--max_iterations=2000`。这是同预算对照，不代表收敛所需轮数。

在 Linux 的 Isaac Lab 训练环境、当前工程根目录执行：

```bash
# 同预算续训对照
python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-11kBaseline \
  --resume --checkpoint checkpoints/model_10999_factorygait.pt \
  --max_iterations 2000 --run_name baseline11k_control

# 肩 pitch 容差试验
python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-ArmEnvelope \
  --resume --checkpoint checkpoints/model_10999_factorygait.pt \
  --max_iterations 2000 --run_name arm_envelope
```

`--checkpoint` 支持现有本地文件路径，优先于云端下载；文件名正则与原 `--load_run` 用法继续支持。需要比较 `Episode_Reward/shoulder_pitch_envelope`、两项 `arm_swing` 以及速度、姿态、滑步奖励，不能只看总 reward。

验证时保留原始 11k、同预算对照和试验模型的关节日志。以下命令先评估原始 11k；将 `--ckpt` 和输出前缀替换为对照或试验 checkpoint 后重复执行：

```bash
mkdir -p logs/arm_eval
python scripts/rsl_rl/sim2sim_play.py \
  --xml resources/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml \
  --ckpt checkpoints/model_10999_factorygait.pt \
  --out logs/arm_eval/baseline11k --no-video \
  --joint-log logs/arm_eval/baseline11k_joints.csv

python scripts/rsl_rl/analyze_arm_log.py \
  logs/arm_eval/baseline11k_joints.csv \
  --output logs/arm_eval/baseline11k_summary.json
```

去掉 `--no-video` 可同时导出视频。关节日志记录实际角度、速度、目标角、末次物理子步力矩、实际足地接触和基座状态。分析脚本默认剔除每段开始的 `1.5 s`（沿用 `sim2sim_percept.py`），分别报告稳态 P10–P90、过渡段/全段峰峰值、参考周期分量和躯干波动。验收时同时比较摆臂和步态，避免把命令切换时的单次抽动当作持续摆臂。
