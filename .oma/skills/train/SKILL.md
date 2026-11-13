# Skill: $train — 八步循环执行段（S4-S8）

**Purpose**: 执行一轮实验的后半循环：**登记 → 云端执行 → 机械判定 → 回填 → 轮转**。前半（S1 检索 / S2 提案 / S3 预注册）由 `$design`/`$tune` 产出任务卡与提案，本 skill 从 S4 接手。
**血源**: loop_agent AGENT_SPEC v0.8（纪律条文移植）+ flux-cli 实战知识（Q3 验证）。
**验证状态**: `[料源实战]`——机器命令（oma decision/resume、poll_task.sh）已 L0/T8 实测；skill 流程本身待 P3 实跑验证。

**Gate in (loop, advisory)**: 任务卡存在（`.oma/cards/{卡名}/TASK_CARD.md`，§0-§6 齐备）+ S2 提案含可证伪 prediction。卡或 prediction 缺失 = 回 `$design`/`$tune` 补，不进入 S4。
**Loop bookkeeping**: 进入时 upsert `.oma/loop.json` → `stage: train`（保持本轮 `exp_id`）。
**Standalone entry**: `oma go train`。

执行细则统一见 [执行、通知与模型交付](execution.md)。以下只保留 S4-S8 的阶段动作；不得绕过该规约用其他轮询器、父权重或旧恢复指令替代权限与交付检查。

---

## Session Resumption Check（每次会话启动，机器命令）

```bash
oma resume          # 扫描 standalone/loop/trajectory orphans/decision_log + lint + flux task info
```

按报告处置（勿手工重复实现——resume 已内置决策表与裁决规则 R-1~R-5）：

| 报告处置 | 动作 |
|---------|------|
| `接续: 重挂后台轮询` | 看护权限允许且无现存 watcher 时 `oma run watch --exp-ref <exp-id>`，同一任务不重建 |
| `paused / held` | 保留暂停，不恢复其他作用域；已提交云训练不变 |
| `notify / verify / collect` | 按执行规约先通知、验证交付；`canEvaluate=true` 后才进 Phase 5 |
| `失败分支 / lost` | 按任务卡 §4 失败分支表处置 |
| `manual（真源损坏 R-4）` | 停止，呼叫人 |
| `clean` | 正常开新轮，从 S4 开始 |

---

## 红线（三条，违反即停并呼叫人）

1. **上真机**：任何真机操作须人批准——唯一保留的执行类红线
2. **reward 结构性新增**：不偏离任务卡 §0 总目标硬约束即可自主提案；越界 = 漂移标记，连续 2 轮越界 → 强制目标复核门（人决策：改目标/回归/收束）
3. **训练预算**：按任务授权范围，启动前足额检查、使用命令级隔离账号（见执行规约）；但核心指标连续 N 轮无实质增益（N 与阈值卡预注册）→ 强制框架反思步，禁止烧算力硬磨

---

## Phase 1 — S4 登记（机器门，无 prediction 不入账）

按 [目标对齐规约](../requirement/goal-alignment.md) 核对 requirements.md 当前版本与任务卡交付锚。登记前解决依赖本轮路线的未决选择；普通新轮不重问已确认目标，首次行为方向待验与性能未达标分别记录。缺新字段的旧实例先从已有指令增量补齐，不将模板升级作为重启任务理由。

1. 复用 design 已分配的 exp_id；尚无编号时分配一次并先写本轮 config（gmTaskId=null）。按 [训练准入](execution.md) 绑定 deployment 范围、实际源码与接口契约，准备 execution.pipeline / plan；登记门不替代后续完整启动门；有界 probe 填实际执行上限，formal 引用同契约的新权重闭环证据。未知不直接投入正式训练。
2. 组织轮次登记对象（`date/exp_ref/change/source/adopted/prediction/note`，无需另存 JSON 文件）：
   - `prediction` 必须多维（速度类 + 质量类至少各一）+ **机制签名**（预期在哪个指标/数据上看到什么）
   - `note` 首行记录 S1 检索命中（lessons/xp 条目 id 或声明"无历史验证支撑"）
   - **目标锚声明**：提案与卡 §0 总目标的关系（集合内推进 / 集合外诊断）
   - `note` 附需求版本、对应目标 ID / 未覆盖项、实际控制分工及方向认可来源 / 首次展示安排；保持现有 CSV 字段，不另造目标账本。
   - note 引用本轮 deployment stage / 契约指纹与待验项；probe 不能在叙述中被写成正式优化已获准。
3. 机器登记（训练准入、追加语义 + lint register 硬门）：
   ```bash
   oma decision register --card {卡名} --file -   # 将上述 JSON 对象传入 stdin
   ```
   准入或 lint 拒绝 → 补齐真实依据 / 修提案重登记，禁止手改 csv；不为绕门填 simulation_only 或重复申请已有用户授权。

## Phase 2 — S5 执行配置（crash-recoverable）

1. 完善 S4 已落盘的同一 `.oma/experiments/{exp-id}/config.json`，不重复分配实验；含显式 `round=rN`、gmTaskId=null、实际框架终端编号 `checkpoint.expectedIteration`。续训登记父权重身份及实际可恢复计数；不从 lap 推算人类轮次
2. 正式训练前核对同运行链最小训练 → 新权重回收 / 加载 → 实际反馈回放的 readiness；环境 / 资产 / SDK / 归一化 / 控制变更按差异重验。核对原生实际生效的观测、动作和执行配置与契约一致，尤其确认 actor / privileged 分组、归一化与部署动作后处理。probe 启动脚本的实际执行上限使用 config 注册的同一字段；formal 不能复用不同契约的 smoke / 闭环结论
3. 平台请求暂存本轮 `.oma/experiments/{exp-id}/work/create-train.json`（下文 `{work}` 指此 work 目录）；模板字段与实际生效参数按项目已验证的 flux 适配配置填写，不每轮复制准备脚本。文件生命周期统一按 [review](review.md)，不把临时请求当证据副本长期保存
4. 提案 diff 必须与登记行 `change` 字段**严格一致**；不改包外内容
5. config 的本轮记录绑定目标版本、覆盖范围和控制方式，与登记 note 一致；目标更新后先核对对本轮协议 / 运行任务的影响，不从“修订目标”自动派生 stop。

## Phase 3 — S6 执行（flux 全链）

先按 [统一准入](execution.md) 核对安装、最小闭环及实际额度来源，写本轮 execution.plan / resource；云任务创建前 `oma run bind-account --exp-ref {exp-id}` 绑定足额账号，再运行 `oma run preflight --exp-ref {exp-id} --json`，非零不 create/run。

适配器将 execution.plan 中已核对的实际请求序列化到 work，使用绑定账号的子进程环境执行 `flux task create --file ... --dry-run`（exit 10 通过）、实际 create 与 run。凭据环境构造复用基座 resources.platformEnvironment；每次子进程独立传 env，不更改父进程 / 全局登录。probe 的真实执行上限必须与 config 的 limit_ref 一致；预检只校验记录，实际提交参数仍由适配器核验。

1. create 成功即回填 config 的 gmTaskId / 必要请求参数，run 前再 preflight；成功启动后记 startedAt 与 trajectory started，核对消费和落盘后清理本次临时输入。run 失败保留已有 taskId 做恢复，不因请求文件整理而重复 create
2. **只读看护**（先检查当前作用域与现存进程）：
   ```bash
   oma run watch --exp-ref {exp-id} --interval 300
   ```
   按 [执行规约](execution.md) 处理终态持久化、实际通知和交付检查；排队 / 部署中不当作结束，不接管外部看护
3. **额度变化 / 计费中断**：先核实实际原因和已创建任务的归属，保存可恢复 checkpoint 与执行进度；重新取得报价、余额 / 占用依据后按执行规约选择足额账号。已有 taskId 不跨账号改绑，新增恢复任务须有当前权限并重新准入；未知不写成额度耗尽，不自动 stop 或重复 create
4. **kill/断线窗口判据**（T8 实测教训）：需人工介入的时点（如验收 kill 测试），先 `flux task logs` 确认出现 iteration 输出再设窗口——status=3 ≠ 实跑

## Phase 4 — 失败分支处置（预注册于卡 §4，通用兜底）

| 症状 | 处置 |
|------|------|
| run 返回余额不足 | Phase 3-3 核实归属与额度，刷新准入后决定恢复 |
| status=6（真 failed） | 存 error.log → 按卡记录执行失败及有效算法观察范围，不自动证伪假设 → 按权限及卡分支 |
| status=6 且 startTime 空 | lost（排队中被平台静默终止）→ 记 lost → 按卡分支重 create |
| 日志含 NaN/OOM/ImportError | 归因分类（nan_explosion/oom/env_bug）→ 回 `$implement`（Bug-Fix Loop） |
| 任务跑完但行为崩 | 仍走 S7 机械判定——"数值过门但行为崩"记 miss 不得过 |

## Phase 5 — S7 机械判定

进入本轮最终判定前，先 `oma run status --exp-ref {exp-id} --json` 核对 `canEvaluate`。绑定任务须结束、已实际通知且当前模型交付通过，并且 evaluation 未暂停；缺新模型是交付问题，不能用旧模型计算行为结果或改记训练未运行。预注册中途诊断保留自身 scope，不改写最终交付门。框架加载与 SDK 交付细节按执行规约的适配器契约处理。

```bash
python tools/play_metrics.py {joints_csv} --baseline {基线csv}   # M1-M9 标准指标
python tools/phase_diag.py {csv}                                  # 归因深挖（相位分解）
```

按 [判定、产物与下一步](review.md) 写本轮 results / verdict：分层判定、数值出处、证据类型、逐门覆盖和接受基线。复用 results.artifacts 引用必要产物，已有 manifest 直接引用，不再复制一套索引。首次行为方向反馈仍按目标对齐规约执行。

## Phase 6 — S8 回填（机器门）+ 沉淀

1. outcome 使用 AGENTS.md 的唯一收束档位表；执行 / 交付 / 评测故障与算法判定分开说明，不能只凭行政中断记算法失败。

2. 机器回填（行级替换 + lint backfill 硬门）：
   ```bash
   oma decision backfill --card {卡名} --exp-ref {exp_id} --outcome {档位} --file -  # stdin: 回填对象
   ```
3. **反事实自查**（note 必答）："若本轮教训在 3 轮前已知，路线会不会不同？"答"会"→ 触发框架重审上报
4. **沉淀钩子**：命中→lesson 候选 / 卡更新 / success_route；**连续多轮后或卡收束时触发 `$consolidate`（不跳过：memory 三表 + 无增益追踪维护）**；卡收束时核算 §7.2 考核（命中率口径见 consolidate skill）
5. 按 [文件生命周期](review.md) 运行 `oma doctor --artifacts --exp-ref {exp_id}`，解决未归类文件、缺失依赖和 work 残留；不将报告另存成逐轮审计文件。真实失败事实可先回填，整理未完成则报告缺项，不宣称文件收尾完成。
6. 有效判定 / 执行收口后追加相应 trajectory 镜像事件；`loop.json` 轮转按当前有效目标、方向反馈与后续授权决定。待用户方向反馈、owner 暂停或仅授权本轮时，记录断点，不自动派生下一轮；普通已授权循环可继续。

7. 登记、判定、暂停或授权变更落盘后，`oma resume --write-summary` 刷新 README / RESUME 的派生区块，保留手写资料；摘要不作为授权真源。

## Completion Signal

```
{任务名} | r{round}（第 {round} 轮） | 判定完成 | {outcome}
  exp {exp_id}  task {taskId}  commit {hash}
  判定: {outcome}（{逐门结果}）
  登记行: decision_log {卡名} 第 {N} 行（lint 两门均过）
  下一步: {卡分支路由}
```
