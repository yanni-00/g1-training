# oh-my-algorithm (OMA) v0.2

OMA 是机器人算法开发的**实验循环编排层**：八步循环（S1 检索 → S2 提案 → S3 预注册 → S4 登记 → S5 配置 → S6 执行 → S7 判定 → S8 回填）+ 机器门 + 会话恢复。
技能文件位于 `.oma/skills/*/SKILL.md`；本文件是**宪法层**：启动协议、门控链、纪律条文、路由合约。

**核心风险**：sim-to-real gap 是每个阶段的首要风险。Reward hacking 是静默失败。切勿仅凭仿真结果断言策略有效。

---

## STARTUP PROTOCOL (run before any action)

1. **安装一致性与机器恢复扫描**（安装后的任务实例）：
   ```bash
   oma doctor --installation --json
   oma resume
   ```
   安装检查未知 / 不一致时，不启动新训练；用指定的同一基座执行 setup。普通安装保留本地文件并报告差异，显式 `--force` 仅替换工具资产，保留需求 / memory / config 业务配置与运行状态；不盲目覆盖本地修改。config.toolchain 记录实际 CLI 源码指纹，不能仅凭版本号或引用路径宣称一致。
   报告含：模式（standalone？）/ 断点（loop.json）/ 任务卡真源状态（lint）/ orphan 任务处置（决策表+flux 仲裁）/ verdict。
   `manual` 结论（真源损坏 R-4）→ 停止呼叫人；其余按报告处置后继续。
2. `.oma/memory.md` → 读取当前有效结论及适用范围；已撤回或被替代的 Dead Ends 不继续约束新决策。
3. 检查请求 skill 的门控条件（Normal Mode 下阻塞即停）。
4. 执行与恢复细则：`.oma/skills/train/execution.md`（暂停范围、统一轮次、终态通知与模型交付；以当前权限决定是否重挂，报告不派生云端 stop）。
5. **团队公共训练经验**：`oma xp index --format md` 读共享 lessons_index，相关才 `oma xp show <id>`；只沉淀算法 / 训练 / 评测 / 控制等技术经验，OMA 编排机制留在基座 docs。路径由 configs/experience.json 或实例 config.experience.lessons_dir 指定，不回退 C 盘个人库。

当前事实以恢复报告与其引用的账本 / 本轮配置 / 结果为准。登记、验收或权限变化后，`oma resume --write-summary` 只读本地真源，刷新 README 与 `.oma/RESUME.md` 的标记区块；手写历史保留，摘要不创建授权。损坏 / 冲突需人工处理时不写摘要，缺层级证据显示未知。

---

## 实验纪律宪法（血源 loop_agent AGENT_SPEC v0.8，全部强制）

### 红线（三条）

1. **上真机须人批准**——唯一执行类红线
2. **reward 结构性新增**：不偏离任务卡 §0 总目标硬约束即可自主提案（仍须副作用预判）；越界 = 漂移标记；**连续 2 轮越界 → 目标复核门**（人决策，禁止默认补丁推进）
3. **按授权范围使用预算，启动前足额检查、按任务隔离账号**；不足先选择已核验的足额账号，未知不估算、不改全局登录；连续无实质增益触发**框架反思**，禁止烧算力硬磨

### 约束撤除类改动（罚项/termination/正则的撤除或削弱）

须附**副作用预判**：该约束当前压制哪些行为、撤除后哪些动作失去成本。无预判不出提案。

### 目标对齐（首次需求与循环内修订）

- 按 `$requirement` 的目标对齐规约执行：正式投入前确认交付含义、首次行为可见时确认方向、中途修订时确认影响范围。先复述可观察行为，只问影响路线 / 验收 / 授权的未决选择；已有明确指令不重复索要确认。
- `.oma/requirements.md` 维护当前目标版本、来源、控制 / 交付边界与逐阶段覆盖表；任务卡 §0 引用同一版本，每轮登记 / config / verdict 绑定本轮范围。论文指标、奖励或代理量不代替用户行为目标。
- 用户反馈区分理解纠正、细节补充、目标变更；agent 误读不能一概归为用户改需求。`LOCKED` 后走增量修订，保留修订前相关文字及适用实验，不重开完整访谈，不回写旧实验为新目标成功。
- 方向认可、量化通过、owner 接受版本、总体收束与真机授权分别记录；已有接受版本保护不覆盖。缺失质量证据或 owner 尚未认可时，不宣称相应主观质量 / 总体接受通过，量化子门仍按冻结标准独立报告；不从目标修订派生任务 stop，不从“可用”派生无限优化。

### 收束语义分级（outcome 唯一合法档位）

| 档位 | 条件 |
|------|------|
| 命中 / 未命中 | 预注册门逐条对照自然结果 |
| 未达标 | 单点结果，附 scope |
| 预算收束 | 止损线触发（**≠证伪**） |
| owner-tolerance收束 | 指标门未全过但 owner 判可用；须 owner 批准 + 未过门清单 |
| 证伪 | 三声明齐备（检验广度 ≥2 种子/组合/混变量分离其一 + 单调性论证 + 混变量排除）——`oma decision` lint 机器强制 |

**总目标未达成的收束必须人批准，AI 上报须带明确倾向（推荐+理由），禁止中立罗列。**

### 判定纪律

1. prediction 多维门控（速度类+质量类至少各一）；无 prediction 不提案（机器拒绝登记）
2. 单变量原则（或预注册声明的功能单元组）
3. 判定必须引用具体指标数值与数据文件（`tools/play_metrics.py` M1-M9 标准口径）
4. 边界判定（量化误差内）透明标注；判定方法变更 note 记理由
5. 存在性/不可能性结论：证据广度不足（<2 种子/组合/混变量分离）只可进 caveat，不得进 lesson 标题

结果层次、事实 / 假设 / 更正、文件生命周期与诊断 / 有限续训的共用细则位于 `.oma/skills/train/review.md`；生成前先复用，临时输入集中本轮 work，收尾用 `oma doctor --artifacts` 检查。复用 results / verdict / memory，不再创建同义账本或按单项排名覆盖接受版本。

### 考核（§7.2，卡收束时机械核算）

- 命中率口径：只核算有效且范围可比的算法轮；prediction 门/分支/机制签名被 outcome 证实=hit；落空或无分支覆盖的意外=miss；混合=0.5；adopted=no 不计分母。行政恢复、接入 / 评测 / 交付故障另记，不凭终判行数混入算法命中率
- 命中 ≥70% 且流程违规=0 → L2 自主维持；<70% 或违规 ≥1 → 降级（提案须人先认可）；<50% → 暂停该任务类新开线
- 跨卡累计，写入卡沉淀 note

### 真源与机器门

| 文件 | 角色 | 写入口 |
|------|------|--------|
| `.oma/cards/{卡}/decision_log_{卡}.csv` | **唯一轮次真源** | 仅 `oma decision register/backfill`（追加+lint 双硬门，禁手改） |
| `.oma/trajectory.jsonl` | 事件流镜像（可重建） | 各 skill append；损坏由 `oma resume --apply` 重建 |
| `.oma/loop.json` | 断点指针 | 进 skill 时 upsert |
| `.oma/experiments/{exp}/config.json` / `results.json` | 本轮执行绑定 / 分层结果及必要产物引用 | train / tune；终态与交付由 oma run 写入 config |
| `README.md` / `.oma/RESUME.md` 的 OMA:CURRENT 区块 | 可丢弃重建的当前视图 | oma resume --write-summary；不作为授权或轮次真源 |
| `.oma/leaderboard.json` / `best.json` | tune 产出 / 部署门 | 仅 `$tune` |

---

## Gate Chain (lifecycle gates + scoped training admission)

```
$requirement → requirements.md + knowledge.md (LOCKED)          ══ HARD GATE ══
┌──────── ITERATION LOOP (one lap ≈ one exp_id = 八步循环一轮) ────────┐
│  $design(S1-S3) → $implement → $train(S4-S8) ⇄ $tune(多轮扫描)      │
│  per lap: hypothesis+prediction+change+outcome（decision_log 行）   │
│  loop state: .oma/loop.json {lap, stage, exp_id, hypothesis}       │
└───────────────────────┬──────────────────────────────────────────┘
                        ▼  ══ HARD GATE: best.json deployGateOpen === true ══
                 人工真机验证（人操作真机；AI 只做数据分析与对齐——walk_diag 模式）
                        → design-feedback（回环）
```

| 转换 | 门 | 类型 |
|------|-----|------|
| 进循环 | requirements.md LOCKED | HARD |
| 循环内互转 | advisory；新轮登记 / 启动仍须按训练准入核对关键部署依赖 | soft，训练准入除外 |
| 新轮登记 | config.deployment 范围 / 契约准入；允许有界 probe，不把登记通过当成启动许可 | HARD；登记机器强制 |
| create / run / retry 前 | 同一 `oma run preflight` 核对安装一致性、实际环境 / 资产 / 新权重闭环及足额账号绑定；细则见执行规约 | HARD；平台调用前复查，不访问云端 |
| 出循环 → 人工真机验证 | best.json deployGateOpen | HARD |

## Stage I/O Contract

| Stage | 读 | 写 |
|-------|-----|-----|
| `$requirement` | paper/访谈 | requirements.md, knowledge.md (LOCKED) |
| `$design` | knowledge / loop seed | design-{id}.md（首轮全量）或 delta（后续轮） |
| `$implement` | design/delta | 代码 + impl/github.json + impl-checklist.md |
| `$train` | 任务卡+提案 | **decision_log 行（机器门）** + experiments/{exp}/ + trajectory 事件 |
| `$tune` | ≥1 train 轮 | leaderboard.json / best.json / tune-current.json |
| 人工真机验证 | best.json 开门 | 人操作真机出 log → **AI 只做数据分析与 sim↔real 对齐**（walk_diag 模式，20260924 实践）→ design-feedback 回环。deploy skill 已删（当前硬件不足以让 AI 控制真机——2026-09-24 用户决定） |
| `$consolidate` | 近期实验 | memory.md（Dead Ends/Working Patterns）+ 考核核算 |
| （归档） | 卡收束 | 卡目录 VERDICT.md（判定记录+验收证据，镜像层，不写真源）——2026-09-24 起 experiment-recording 移除，归档职责并入卡三件套 |

## Keyword → Skill Routing

| 关键词 | Skill |
|--------|-------|
| requirement / clarify / success criteria / 目标补充 / 理解纠正 / 修改目标 | `$requirement`（已有目标走增量修订） |
| design / reward design / policy / network | `$design` |
| implement / build / write trainer | `$implement` |
| train / launch experiment / 开跑 | `$train` |
| tune / sweep / ablation / 调参 | `$tune` |
| 分析 exp / 对比 / joints / leg symmetry / yaw | experiment-analysis（→ tools/play_metrics M1-M9） |
| 记录 exp / 归档 / 复盘 | 卡目录 VERDICT.md（判定记录+验收证据）+ `$consolidate`（沉淀 memory） |
| deploy / sim2real / 真机 / 对齐真机数据 | 人工真机验证：人操作真机 → AI 分析 log（experiment-analysis 口径）；反馈走 design-feedback |
| consolidate / 沉淀 / update memory | `$consolidate` |
| resume / 恢复 / 断线 / kill / 暂停 | train 执行规约 + `oma resume` / `oma run`（按作用域） |
| gm / flux / gradmotion / 平台 | gradmotion skill（flux 实战版） |
| 下一轮 / next lap / 再改 | bump loop.json lap → 路由进 $design/$implement/$train/$tune |

## Standalone Mode

`.oma/standalone.json` 存在 OR `oma go <stage>` → 阶段门 advisory 化（显示缺失清单后继续）；不豁免训练前部署契约准入。`oma go off` 恢复门控。

## State Files

| 路径 | Owner |
|------|-------|
| `.oma/requirements.md` / `knowledge.md` | `$requirement`（LOCKED） |
| `.oma/designs/design-{id}.md` | `$design` |
| `.oma/impl/` | `$implement` |
| `.oma/cards/{卡}/TASK_CARD.md + decision_log_{卡}.csv + VERDICT.md` | 任务卡三件套（真源所在） |
| `.oma/experiments/exp-{id}/` | `$train`/`$tune`（config.json create 前写盘） |
| `.oma/leaderboard.json` / `best.json` | `$tune` |
| `.oma/trajectory.jsonl` | 各 skill append（镜像） |
| `.oma/loop.json` / `tune-current.json` / `standalone.json` | 指针类 |
| `.oma/memory.md` | `$consolidate` only |
| 配置的团队 `experience/lessons/` 与同仓库 `lessons_index.json` | 公共训练经验；沿用团队生命周期 / 分类 / 置信度，不建立第二套 XP 索引 |

---

<!-- OMA:USER-OVERLAY:BEGIN -->
# F1 全身行走 — 本次用户授权

来源：2026-11-06 当前聊天。

- 使用本项目 oh-my-algorithm 基座，新建独立 Loop。训练实现位于 ../../6_sim-train/F1-29DOF/sim_walking/，状态与需求保留在本实例。
- 从本地 G1 unitree_rl_lab 原始 velocity 路线适配；unitree_rl_lab_train 的历史调整仅作后续候选，不混入初始奖励。
- 效果基准由用户指定的 1_G1_deploy / yanni-G1-walk-01 实际模型决定，必须核实 FSM、清单及文件哈希，不凭文件名推断。
- 使用 SYS-SPEC-0002-V1.1-F1整机模型29DOF，保留源 URDF 与 meshes；不将旧 F1/X1 辨识结果视为新硬件参数。
- 当前只构建 Loop、训练代码与本地验证。不启动训练、云端 probe、调参、自动监控或远端发布；没有本次算力预算与账号授权。
- 用户负责真机，agent 负责训练、仿真评测、sim2sim 与离线交付；本次搭建不代表行为、部署或最终目标已通过。
- 初期允许手臂摆动不明显；最终要求双臂交替、与对侧腿配合，同时保留行走质量。首次有效行为需提交指令、模型身份和视频供用户判断。
- 实现阶段已有用户提供的参考代码路径；不重复询问是否有参考。通用 skill 的 push、动态 smoke 与云端门在本次授权外，保留待验，不为完成模板而执行。
- 当前不登记虚构轮次、不生成 best.json 或训练结果；后续正式运行先按 OMA 原生登记及 preflight 核对环境、资产、控制链、模型交付与预算。

## 2026-11-13 当前授权覆盖
用户已指定全身 PD、URDF friction=0/damping=0.05，暂不加其他辨识，授权按实际余额选择账号并向 GM 服务认证、查询和训练，授权 GitHub 新分支。旧的仅搭建限制不再适用于本次首版训练。真机仍用户负责，不无限续烧账号池；首次有界接入后才进同链仿真优化。
<!-- OMA:USER-OVERLAY:END -->
