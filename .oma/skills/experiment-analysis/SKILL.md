# Skill: experiment-analysis — M1-M9 标准指标分析（单实验/对比）

**Purpose**: 对 play 产物 joints csv 计算标准指标并机械对照——S7 判定的唯一数据出口。
**工具（vendored，实战血统）**: `tools/play_metrics.py`（M1-M9，METRICS 口径唯一实现）、`tools/phase_diag.py`（相位归因深挖）。
**验证状态**: `[料源实战]`（cut00-07 + 0921 walk_diag 对齐分析全程使用）。

## 触发

- 「分析 exp_XXX」 / 「对比 exp_A exp_B」 / 「按卡判定 {exp_id}」

## 步骤 1：定位数据

| 来源 | 路径 | 说明 |
|------|------|------|
| 循环产物 | `.oma/experiments/{exp-id}/` 下 `*_joints.csv` | 列含 step,time_s,phase,base_*,cmd_x,foot_fz_*,gait_state,L*/R*_{des,act,err} |
| 基线 | 任务卡 §0 引用的 baseline csv | 判定必带 |

- 同 exp 多 csv → 取最新或问用户；**禁止混协议**（恒定 vs 调度 play 不可直接比）

## 步骤 2：计算指标

```bash
python tools/play_metrics.py {csv} --baseline {基线csv}
```

输出（口径与 METRICS 规范严格一致）：
- **M1** 分段速度跟踪率（R1 按 cmd 分段，禁混段）
- **M2** 步频 duty（R3 行走段 flight 占比×2）
- **M3** 支撑对称比（R4 fz>50N 判支撑）
- **M4** 关节跟踪 MAE（度，逐关节+双腿均值，重点关注超基线 1.5× 的关节）
- **M5** 姿态包络（pitch/roll absmax）
- **M6** stand yaw 漂移（R2 真 stand 段；仅观察项，sim/real 脱钩）
- **M7** flight 占比 | **M8** stop 双支撑占比 | **M9** 行进 yaw 漂移率（先 unwrap）

## 步骤 3：机械判定（按卡 §2）

1. 逐门报告"当前值 vs 基线值 vs 差值"三列
2. 判定引用**必须带数据文件名**（数值纪律）
3. 边界判定（差距在量化误差内）→ 注明误差量级与判定倾向
4. 判定方法偏离卡面口径 → 记入该轮 note（先例：episode→duty）

## 步骤 4：归因深挖（判定未命中或需机制证据时）

```bash
python tools/phase_diag.py {csv}
```

相位分解定位到具体关节的 des/act 行为（如 hr des 爆炸 = policy 主动命令，非执行误差）。

## 已知陷阱清单（判定前必读）

- 调度 play 的 cmd_x=0 混有 rotate 段 → 用 R2 隔离真 stand
- 低抬脚 flight 碎片化 → episode 计数失真，用 M2
- 单点触地冲击（fz max 数千 N）→ 看 R4 均值而非 max
- csv 中 des/act/err 单位为度，勿二次换算
- yaw 漂移必须先 unwrap（Euler ±180° 卷绕误读）

## 约束

- 只输出客观指标与有数据支撑的解读；机制推断、规划下一 exp 须用户确认后进行
- 多实验对比仅同系列同协议；行数差异大注明时长偏差
- 旧版引用的 `locomotion-lab/analysis_scripts/compare_*.py` 路径**已废弃**（v0.1 假路径，审核结论）——一切以本 skill 的 vendored 工具为准
