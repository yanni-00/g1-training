# F1 全身行走 Requirements
Status: LOCKED
Goal version: F1-FULLBODY-WALK-v2 (current); v1 retained below as historical scaffold scope
Framing: A — 用户指定原始方法的平台适配。
Confirmed by: 2026-11-06 当前聊天原始需求及两项追问回复。

## 交付目标与范围
本次构建独立 OMA Loop、F1 全身 velocity 训练代码、资产与控制契约、参数接口、回放/导出入口，并做不训练的本地验证。
后续速度指令驱动新 F1 全身行走，达到指定 G1 部署模型效果；初期允许摆臂不明显，最终双臂交替且与对侧腿配合，同时保持行走质量。
用户负责真机；agent 负责其他训练、仿真评测、sim2sim 与离线模型交付。当前只授权本地搭建，不含训练/probe/云资源/自动化/远端发布。

## 来源与目标基准
方法：../../6_sim-train/G1/unitree_rl_lab/ 原始 locomotion velocity + RSL-RL PPO；训练副本 unitree_rl_lab_train 的中途改动只作后续候选。
效果基准：用户指定 ../../1_G1_deploy/ 的 yanni-G1-walk-01，FSM.Velocity.type=Velocity11k，policy_dir=config/policy/velocity/11k。
该部署包 manifest：checkpoint_iter=10999，source_task=TASK_20261020_069，training_commit=5421b4f；checkpoint_sha256=8d4815ac2754bdf249d24e88c68ad1707679d09a2b2f56cb56916e86184178db。核验见 .oma/impl/baseline_identity.json。
F1 资产：../../f1_simulation_ssot/01_SPEC_规范与接口/SYS-SPEC-0002-V1.1-F1整机模型29DOF/。代码及资产在 ../../6_sim-train/F1-29DOF/sim_walking/；原始哈希见 configs/source_manifest.json。
现有 f1-walking 是固定上肢的另一条路线，本次独立建立。

## 目标覆盖表
| 目标 | 协议 | 本次状态 |
|---|---|---|
| OMA 同源安装与训练基座 | doctor、源码差量、静态/CPU 检查 | 本次交付 |
| 全身开放，关节/姿态/限位正确 | URDF、FK、STL、选择器和运行时顺序检查 | 静态验证；引擎待验 |
| 持续行走、起停、侧移与转向 | 后续同指令视频、速度/姿态/足接触日志 | 未训练、待验 |
| 达到指定 G1 效果 | 固定模型、同协议 G1/F1 对比；登记前冻结阈值 | 待验，无虚构数值门 |
| 双臂与对侧腿配合 | 手肘相对躯干运动、左右一致性、实际臂腿相位与视频 | 最终目标；首轮不添加摆臂奖励 |
| sim2sim 与新权重离线交付 | 实际 F1 运行契约、模型哈希与连续闭环 | 入口准备；动态待验 |
| 真机 | 用户自己推进 | agent 不操作 |

## 平台证据与未决项
关节、限位、力矩/速度、被动动力学以指定 URDF 为准。PD/armature 由 G1 原始配置作有出处的仿真工程起点，不是 F1 辨识结果。新 F1 完整辨识、实际传感器标定、部署时序/动作保护链尚无本实例证据，未知保持 null。
下肢保留 G1 膝弯曲量，髋踝由 F1 FK 求平足；上肢以 G1 姿态幅度按 F1 世界系镜像方向适配；额外肘 yaw 与腕取 URDF 零位。高度由真实 collision STL 推导，全部是待动态验证初始化。
源上肢多数 link 缺 collision；保留原资产，现有检查不能证明全身碰撞覆盖。摆臂阶段需补足此缺项。
当前没有 F1 动态训练、新权重闭环、方向接受或总体成功证据。账号/预算/轮数未授权，原代码默认值不代表本次运行许可。

## 方向验收与版本记录
首个有效 F1 行走产物出现后，向用户提供实际指令、模型/控制身份及视频确认方向。CPU forward、编译或 reward 不代替行为验收。
F1-FULLBODY-WALK-v1：当前指令确认方法、资产、基准和分工，尚未接受任何 F1 行为版本。

## 后续补充 — 2026-11-06
用户明确：参数辨识等待后期测试完后注入。本次完成接口与代码，不等待辨识、不填造结果；来源为空的字段保持 null，工程种子配置不标为辨识。

## v2 增量修订 — 2026-11-13（覆盖上述 v1 当前仅搭建限制）
来源：用户本轮 PD 文本与附图、开始一版训练指令、按实际余额选账号回复，以及 GitHub 新分支指令。
当前交付：先完成同配置有界真实接入，再启动首版从零训练并回收/评估新 F1 权重；效果目标仍是已冻结的 G1 11k，首版允许弱摆臂。真机分工不变。不是无限调参授权。
PD 与被动参数唯一当前输入为训练目录 configs/owner_training_profile.json，附图副本 .oma/impl/owner_pd_20261113.png。左右下肢同值，顺序 hip_pitch/hip_roll/hip_yaw/knee_pitch/ankle_pitch/ankle_roll；上身 waist/lumbar 及双臂按附图。
用户要求暂不加其他辨识：不注入旧辨识、附加 armature、摩擦补偿、交付 LPF 或延迟。附加 armature=0 是本轮不施加该项的工程设置，不是已辨识零值；辨识文件仍 pending/null。
只改本 Loop 训练副本的可动关节 dynamics，friction=0、damping=0.05；F1 SSOT 原件保持原样。PD kd 是导数反馈项，被动 damping 是独立物理项；当前 implicit drive 合计二者，运行时核对。
授权创建 GitHub 训练仓库独立分支，按实际余额/占用选择足额账号，不改全局登录或其他任务；预算上限由所选账号已核验可用额度限定，不借账号池总额派生无限运行。没有实际 taskId 前不宣称开训。
旧 v1 无训练结果，不追溯改判。首个有效行走视频仍需用户确认方向。

当前 r1 范围为用户分工内的仿真接入及首版策略训练准备；配置中 simulation_only 仅表示本轮无硬件驱动/真机验收，不改变长期行走与最终摆臂目标，不宣称部署通过。正式仿真优化须先验证同运行链的新权重交付、加载与反馈回放。
