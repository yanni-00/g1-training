# Skill: $tune — 预注册扫描循环（多轮八步 + 部署门）

**Purpose**: 系统性超参/配方扫描 = **每轮完整走八步循环 S4-S8 的连跑**（提案→预注册→登记→执行→机械判定→回填），累积 leaderboard，最终产出 `best.json` 部署门判定。
**血源**: loop_agent AGENT_SPEC v0.8（单变量/预注册/收束语义/考核）+ OMA v0.1 tune 骨架（leaderboard/best.json 保留）。
**验证状态**: `[料源实战]`（各机器命令已实测）；skill 全流程待 P3 实跑验证。

**Gate in (loop, advisory)**: 任务卡 §0 总目标锚 + 消融/扫描计划（卡 §2 或 design 的 ablation plan）。理想输入 ≥1 条 train 轮已完成；循环内不阻塞。
**Loop bookkeeping**: 进入 upsert `.oma/loop.json` → `stage: tune`；写 `.oma/tune-current.json` 会话指针（consolidate 完成才删）。
**Standalone entry**: `oma go tune`。

---

## Session Resumption Check（每次会话启动，机器命令）

```bash
oma resume
```

- `tune-current.json` 存在 → 恢复该 tune 会话（phase 字段路由：sweep→查 activeExpIds 各任务 / final_eval→收结果 / consolidate→触发）
- orphan 处置同 $train（决策表/失败分支表由 resume 内置）
- 判定 / 证据、产物保留、可比轮次和基线替换采用 [train 判定规约](../train/review.md)，不为 sweep 复制一套阈值或状态。
- 暂停 / 恢复、轮次、终态通知及模型交付统一采用 [train 执行规约](../train/execution.md)，不为 sweep 另建看护或通知状态；各变体保留自己的 exp / task 绑定，暂停后不自动开新轮

---

## 扫描纪律（预注册，动手前写死）

执行 [目标对齐规约](../requirement/goal-alignment.md)：扫描计划引用当前需求版本、已认可行为方向及覆盖目标。若首次方向尚未确认，先给有限行为证据并等待相应反馈，不因用户未回复启动依赖该选择的批量优化；已有同目标 / 控制版本认可直接引用。收到目标补充 / 纠正时增量审查受影响变体，独立已授权工作可继续，不能自动停止云任务或重问同一授权。

1. **单变量原则**：每轮 config 与基线只有一个变量差异（或卡面预声明的"增量单变量组"——组内字段须在 S3 声明共同构成一个功能单元）
2. **Dead Ends 排除**：`.oma/memory.md` Dead Ends 表 + 卡面历史失败分支——**不得重复已证死路**（无新实验理由）
3. **sweep 计划预注册**：动手前写 `.oma/tune-{YYYYMMDD}-{seq}-plan.json`（策略/基线/hypothesis/全部变体配置/GPU 预算），并立即写 `tune-current.json`
4. 每轮走 $train 的 S4-S8 全流程（**含 lint 双门**），leaderboard 逐轮追加

| 策略 | 适用 |
|------|------|
| sequential_ablation | 变量近独立，逐个改 |
| grid sweep | ≤3 变量且疑有交互 |
| directed search | memory.md Open Hypotheses 有明确假设，窄域聚焦 |

## Phase 1 — 计划（S1-S3 for sweep）

1. S1 检索：`oma xp index` + lessons（命中写每轮登记 note）
2. S2 假设：每个变体一条可证伪 prediction（多维门）+ 目标锚声明
3. S3 预注册：全部变体的门与失败分支（写在 plan.json 与卡上）
4. **框架反思触发器**：核心指标连续 N 轮无实质增益（N/阈值卡预注册）→ 停止扫描，逐条列当前搜索空间隐含假设逐条检验，产出新框架候选后方可继续

## Phase 2 — 执行（每变体一轮完整循环）

按 `$train` Phase 1-6 逐轮执行（S4 登记→S5 配置→S6 flux+轮询→S7 判定→S8 回填）。
并行按任务授权和实际资源许可，各写自己的 `exp-{id}/`；每个任务独立完成执行规约的安装 / 运行链 / 额度准入并绑定账号，核对共享账号占用，不改全局登录。
失败按任务卡分支处理，其余已授权变体可独立继续；技术故障与有效算法判定分开，不机械把每次中断计成算法未命中。

## Phase 3 — 收集与机械分析

1. 读全部新 `results.json`，逐变体 `metric ± std`
2. 按任务卡冻结的改善 / 保护门比较；缺统计依据、协议不兼容或有明显取舍则保留不确定，不默认使用统一 std 倍数宣布胜出
3. 写 `.oma/tune-{tuneId}-analysis.md`（结果表/胜者/负者/观察模式/死路候选）
4. leaderboard.json 追加（按 metric 排序，append-only）

## Phase 4 — 停止条件（任一即转终判）

1. 同目标 / 协议可比轮触发任务卡预注册的无增益 / 平台期标准；标准或证据缺失先补依据，不用统一阈值
2. 任务卡明确的止损 / 有限执行范围到点，或本阶段冻结门达到；不自动等同总体完成
3. 消融计划全变量扫完
4. 用户说 evaluate / done
5. 框架反思触发器命中（回 Phase 1 重列假设）

## Phase 5 — 终判与部署门

按 [判定规约](../train/review.md) 分开判阶段覆盖、用户接受与部署门，候选按冻结完整比较条件晋升；保护门失败、证据不确定或视觉相近时保留原接受基线 / 两版。
1. 最优候选按卡 §6 play 协议跑终判（训练 / 评估种子分栏，有实际样本才报告 mean±std；**双线验收**：指标线 + 业务代理线，两线都必须 S3 预注册，禁事后反推）
2. 写 `experiments/{best-exp}/eval-results.json`
3. **写 `.oma/best.json`**（仅本 skill 可写）：
   ```json
   { "expId": "...", "primaryMetric": {"mean": null, "std": null},
     "allThresholdsPassed": false, "deployGateOpen": false,
     "unpassedGates": [], "configRef": "...", "designRef": "..." }
   ```
   - 指标门未全过但业务代理线过 → **不得自行开部署门**：生成 owner-tolerance 收束提案（两选项：A tolerance 收束转下阶段 / B 继续调参 1 轮，各带预期）上报人裁定
4. trajectory 追加 eval_complete 镜像事件；同步 `oma resume --write-summary`，不手写重复当前状态

## Phase 6 — 考核核算（卡收束时，§7.2 口径）

1. 回溯本卡已终判的有效算法轮并核对可比范围；行政恢复及技术故障另记，不凭终判行数计入命中率。计分只引用 AGENTS.md §7.2，保留纳入 / 排除的实验 ID 与理由
2. 命中率 + 流程违规（八步任一环未走机器门）写入卡沉淀 note
3. 命中 <70% 或违规 ≥1 → 降级：提案恢复"S4 登记前 change 文本须人先认可"（连续 2 卡达标后恢复自主）
4. 立即触发 `$consolidate`（不跳过）；完成后删 `tune-current.json`

## Completion Signal

```
{任务名} | r{round}（第 {round} 轮） | 扫描终判 | {状态}
  Tune ID: {tuneId}  |  Sweep: {N} 轮，胜者 {exp-id}（{metric}）
  终判: {mean}±{std}  |  门: {逐条 ✓/✗}
  部署门: {open / blocked——原因+建议}
  考核: 本卡累计命中 {x}/{y}  |  Memory 更新: $consolidate
```
