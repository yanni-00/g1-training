# F1 BeyondMimic 训练代码

独立命名空间 `f1_beyondmimic`，使用项目 G1 BeyondMimic 路线的跟踪 MDP、PPO 与命令条件循环参考。参考关节目标驱动 actor，critic/reward 可使用真值；actor 不包含实际位置或线速度估计。
2026-11-16 用户已授权采用 f1-29dof-walking 的用户 PD 和 URDF 动力学基线，使用独立分支和 GM 正式训练，每任务上限 50 credit，不另提交 100 轮试跑。控制基线已经同步，辨识仍 pending；训练输入另存在 `data/prepared/owner-pd-20261116/`，原交付数据保留。正式任务中的原生审计、权重与行为结果按实际执行回填，尚不声称已训练或部署通过。

2026-11-09 数据交付见 [数据说明](data/regenerated/20261109/final/README.md)，周期参考为 `data/regenerated/20261109/final/walking_cycle_50hz.npz`。起步、连续行走和停步另有 qpos 与原生全身 FK 文件；原始数据保留。

## 文件

| 路径 | 用途 |
|---|---|
| `source/f1_beyondmimic/` | 原生任务、F1 资产配置、PPO、动作/观测/参考和 checkpoint 契约 |
| `resources/f1/` | 指定 SSOT 的原始 URDF / meshes，按字节保留 |
| `configs/robot_contract.json` | 模型关节/限位/静态姿态；不代表 SDK 或引擎顺序 |
| `configs/parameter_identification.json` | 待注入的真实硬件参数，无 G1 PD 兜底 |
| `configs/deployment_contract.json` | 真实 F1 观测、时序、保护和状态机待接入 |
| `data/motion_manifest.json` | 原数据/候选/重生成数据清单与哈希；prepared 训练准入仍待确认 |
| `tools/regenerate_ssot_walking.py` | 本地完整重生成入口；只处理离线数据 |
| `configs/ssot_motion_sources.json` / `ssot_motion_processing.json` | 源动作映射、修正和周期数值设置 |
| `configs/reference/` | G1 方法快照及稳定 checkpoint 实际配置，仅作来源 |
| `configs/evaluation.json` | 本版行为范围，时间表/阈值待实际数据后冻结 |

URDF 被动阻尼和控制器 D 分别记录；导入隐式驱动时使用其明确合成值，不将总阻尼再次重复相加。动作幅度公式沿用源 BeyondMimic 配置，代入后续实际 F1 参数。原生动作使用 F1 硬限位；它尚不代表已匹配实际 F1 部署保护链。
数据必须带 joint_names/body_names；原生构造按实际引擎名称核对并重排，名称缺失或重复时拒绝。新的 `cyclic_heading_stride_local_v1` 保留相对匀速路径的周期骨盆摆动，位置与速度使用同一坐标定义。支撑停止和周期 wrap 不重置机器人。首版明确拒绝后退/侧移和未验证的原地转向。

## 本地验证

在本目录运行：

```powershell
$retargetPython = '.runtime/retarget-py312/Scripts/python.exe'
& $retargetPython scripts/train.py --check
& $retargetPython scripts/evaluate.py --check
& $retargetPython tools/check_scaffold.py --output ../.oma/impl/validation.json
```

CPU 测试使用临时合成输入/临时模型，只检验软件契约；不产生可用策略或运动学/行为验收结论。

## 后续运行接口

项目隔离环境已有 MuJoCo，用于真实数据的离线 FK、碰撞查询和运动学回放；Isaac Lab 实际加载与动态策略跟踪未执行。环境依赖见 `configs/ssot_retarget_requirements.lock`。Python 安装器沿用了已有的 `D:/pycharm-community-2022.3.1/` 安装并更新为 3.12.10，项目依赖另装在 `.runtime/retarget-py312/`；原默认 3.10 未替换。
真实运行前：核验候选数据在实际 Isaac 资产/原生运行时的对应关系；注入有来源的逐关节参数；绑定 F1 实际部署/控制链；冻结评估协议；获得新的运行授权及 OMA 轮次/账号/预算/preflight。不要仅修改几个 enabled 字段便开始训练。

后续获准时入口为：

```bash
python scripts/train.py --headless
python scripts/train.py --headless --checkpoint /absolute/path/to/F1/model.pt
python scripts/evaluate.py --headless --checkpoint /absolute/path/to/F1/model.pt --output /absolute/path/to/new-evaluation --video
python scripts/export.py --checkpoint /absolute/path/to/F1/model.pt --output /absolute/path/to/new-package
```

训练规模/seed 从明确授权的 launch.json 读取，无隐含 latest、G1 权重恢复或自动云端提交。保存实际 runtime 与归一化，首次保存验证原生 actor 与离线 actor 一致性。
回放输出完整 states.csv、阶段误差 metrics.json 和可选视频；首次 done 立即结束，CSV 保存 pre-step 状态。足端刚体速度只标为接触滑移代理量，不能代替真实足底接触面滑移。阈值与人工视频验收未完成时，behavior_acceptance 保持 false。
导出保留归一化 actor.pt 与 runtime.pt 的身份绑定，拒绝覆盖既有包；离线导出不开放真机执行门。
