# F1 配置与来源

| 文件 | 用途与来源 |
|---|---|
| `robot_contract.json` | 指定 URDF、模型关节顺序、限位/轴向、FK 初始姿态、真实 STL 足底高度，以及原始 G1 obs/action/时序 |
| `control_profile.json` | G1 同功能关节 PD/armature 工程起点；每项含 source_joint/source，不是 F1 辨识 |
| `parameter_identification.json` | 新 F1 后期测试后注入；初始 pending/null，不从旧 X1 或下肢 Loop 抄结果 |
| `launch.json` | 当前本地搭建范围，training/evaluation 均关闭；不启动算力或仿真任务 |
| `source_manifest.json` | 原始代码、F1 URDF/引用网格与副本哈希；adapted_sha256 为交付源码快照 |
| `reference/` | 原始 G1 配置快照、原 G1 足部几何与指定部署模型清单；只作来源，不注册 G1 训练任务 |
| `eval_schedule.json` | 指定 G1 11k 的真实回放 CSV 命令序列；未来用于相同工况对比，不是 F1 达标结果 |

`model_joint_names` 是本训练模型顺序，**不是新 F1 硬件 SDK 顺序**。每个关节由名称显式选定；Isaac 导入器实际枚举顺序在环境创建后另行核对并映射。禁止以维度相同当作语义相同。

姿态计算：保留原始 G1 的膝弯曲量，使用新 F1 FK 求髋踝平足；肩 pitch/肘 pitch 继承原 G1 幅度，肩 roll 按 F1 世界镜像方向取负；额外肘 yaw 和腕用 URDF 零位。基座高度为初始姿态下真实足底 collision STL 最低点到基座的距离。足部 clearance 保留原 G1 足底上方的目标量，按两种模型的实际 sole offset 换算到 F1 ankle link 原点。都是初始化工程设置，尚无动态效果证据。

原始 URDF 不改动。F1 力矩和速度上限取实际 URDF；PhysX implicit drive 的 damping 配置为继承的 PD derivative 加 URDF 被动 damping，不额外再施加粘性摩擦。原 URDF friction 保留；其导入器/后端实际语义需要未来引擎加载时核验。全身碰撞覆盖仍以原资产为限，上肢缺少 collision 的 link 不会因 self_collisions 开启而获得碰撞体。

参数测试完成后：

1. 保留本文件的 `urdf_sha256`，将真实测试/报告文件以 `{ "path": "相对训练根目录或绝对路径", "sha256": "实际文件哈希" }` 记入 `sources`。
2. 按真实关节名填入已确认的 `stiffness`、`pd_damping`、`armature`。仅有部分关节/字段时设 `status=partial`，其余明确沿用工程起点；只有所有字段齐全且核验后才设 `validated`。null 不转换为零。
3. 本接口只注入这些已定义参数。摩擦、延迟、执行器模型的辨识若涉及其他物理语义，须先适配对应实际控制路径，不可只改状态字。
4. `effective_control()` 会核对模型绑定、来源文件哈希、关节集合和数值。更改控制配置会使旧 checkpoint 绑定失效，不能悄悄续训或回放。
5. 参数就绪本身不启动训练。未来按实际授权登记 OMA 轮次，完成原生 preflight 和有界真实接入，再配置本轮运行入口。

`tools/build_contract.py` 用于重建静态契约和初始工程 profile；会重建 `control_profile.json`，但不会覆盖已存在的辨识文件。已有实验后不能无审阅重建并让历史侧车重新自签。

## 当前启用版本：2026-11-13 owner baseline（覆盖前述 G1 工程起点）
当前 control_profile.json 从 owner_training_profile.json 生成。下肢按 hip_pitch/hip_roll/hip_yaw/knee_pitch/ankle_pitch/ankle_roll，双侧同值；上身按附图。原始来源保存在 OMA 用户输入快照。
训练 URDF 可动关节 friction=0、damping=0.05，SSOT 不变。额外 armature=0（本轮不施加该项，不是辨识结果），无额外摩擦补偿/LPF/辨识。PD kd 与被动0.05分开记录，implicit 总阻尼为二者之和。
GitHub 分支用无损 tar.gz 传递资产，入口恢复后逐文件校验；不简化网格或更换机器人。GM 返回单独 PT 时，文件内的 f1_contract 用于校验，不依赖额外 sidecar 成功上传。
