# Skill: $implement

**Purpose**: Turn the approved robot algorithm design into working code — including the RL environment, reward function, observation/action wrappers, policy network, and training entry point.
**验证状态**: `[料源设计→P3 验证]`。**Smoke test 暂缓**（2026-09-24 用户决定）：当前算力不适合试跑（本机无 Isaac Gym），待算力适配后启用；暂缓期间首个 GM 云端任务（≤100 iter 量级）承担冒烟职责，训练失败走 Bug-Fix Re-entry 兜底。

Two paths — selected automatically:
- **Path A — Adapt**: A reference codebase is registered at `.oma/codebase/config.json`. Read the existing code, identify what needs to change, implement directly.
- **Path B — Scratch**: No reference codebase. Scaffold and build from the design spec.

**Earliest start**: After `$requirement` completes. Can start reading the codebase (Path A) in parallel with `$design`, but writing code requires the finalized `design-{id}.md`.

**Gate in (loop, advisory)**: a current `design-{id}.md` is the ideal input, but inside the iteration loop `$implement` is **not blocked** by its absence — the only hard requirement is `requirements.md`/`knowledge.md` LOCKED. For a **delta lap**, take the change straight from `.oma/loop.json` `hypothesis` and the lap's `experiment.json` (no fresh design doc needed). A `$train` that surfaced a code bug re-enters here directly (Bug-Fix Re-entry).
**Loop bookkeeping**: on entry, upsert `.oma/loop.json` → `stage: implement` (keep the lap's `exp_id`).
**Standalone entry**: Allowed via `oma go implement`. If design is missing, ask user to paste the key design decisions (obs space, action space, reward terms, network arch, RL algorithm) inline.
**Gate out**: `.oma/impl/impl-checklist.md` with evidence for completed items and code pushed to GitHub. This is implementation readiness; entering training also requires the shared deployment preflight below. Advisory loop entry does not waive that admission.

---

## Phase 0 — Context Loading

按 [目标对齐规约](../requirement/goal-alignment.md) 核对当前需求版本、控制输入 / 输出分工与本轮覆盖目标；实现检查单引用这些信息。论文示例或既有脚本不能代替交付要求（如固定轨迹播放不能冒充指令闭环）。路线关键歧义先聚焦澄清，已有明确授权直接沿用；普通实现细节自主处理。

为首次行为展示准备可识别的模型 / 控制方式、实际输入与视频 / 行为记录，优先复用已有产物。编译、接口或 smoke 通过只证明对应范围，不标为用户方向或自然度通过；收到补充 / 纠正按 `$requirement` 增量路径处理受影响部分。

文件生成与临时脚本遵守 [文件生命周期](../train/review.md)：重复操作参数化为项目工具，一次性辅助脚本暂存本轮 work；不按轮复制整套 prepare / finalize / recovery 代码。

按 [训练前部署契约](../train/execution.md) 将已有部署源码反向映射到原生训练实际配置，并把关键未知写入现有 impl-checklist、绑定 config.deployment；不把估计器可用性 / 动作限幅 / 时序 / 状态机等关键依赖留到正式训练后才验证。先有界接入、最小新权重尽早跑实际部署离线闭环；静态检查或模型 forward 通过不等于此门通过。

Read these files before doing anything else:

| File | Status | What to extract |
|------|--------|-----------------|
| `.oma/requirements.md` | required | Task, metric, framework, compute budget, seed count N |
| `.oma/designs/design-{id}.md` | required | Architecture spec, loss, optimizer defaults, ablation variables |
| `.oma/memory.md` | if exists | Dead Ends — do NOT reimplement these |
| `.oma/codebase/config.json` | if exists | `srcPath` — the reference repo root (pre-registered via `oma index`) |

---

### Step 0 — 询问是否有开源参考实现（必须执行，不可跳过）

**在选择路径之前，先问用户一个问题**：

> "实现阶段开始。请问你是否有可以参考或复用的开源代码库？
> - **有**：请提供本地路径（如 `./legged_gym`）或 Git 仓库地址，我会在其基础上做改造（Path A — Adapt）
> - **没有**：我将从零开始按设计文档实现（Path B — Scratch）"

**等待用户回答后再继续**。不要假设、不要跳过这个问题。

**根据用户回答选择路径：**

| 用户回答 | 路径 | 操作 |
|---------|------|------|
| 提供了本地路径 | **Path A** | 如果 `.oma/codebase/config.json` 尚不存在，提示用户执行 `oma index --src <path>`，等执行完毕后读取 `config.json` 继续 |
| 提供了 Git 地址 | **Path A** | 提示用户先 clone 到本地再运行 `oma index --src <local-path>`，等执行完毕后继续 |
| `.oma/codebase/config.json` 已存在 | **Path A** | 直接使用已注册的路径，告知用户："检测到已注册参考代码库：`{srcPath}`，将基于此做改造" |
| 明确说"没有" | **Path B** | 直接进入 Path B |

> **注意**：如果 `.oma/codebase/config.json` 已存在但用户说"没有开源实现"，以用户的回答为准，走 Path B，并提示用户可以用 `oma index --src <path>` 重新注册。

---

### Path 选择后的检查

**Path A 选定后**：确认 `srcPath` 目录可访问，Codex 能读取其中的文件。

**Path B 选定后**：确认 `design-{id}.md` 中的架构描述足够详细（有层数、维度、loss 公式），若不够详细先从 design 补充，不要用模糊描述写代码。

If design is not yet finalized and the user triggers `$implement` early (Path A only):
→ Read the codebase to understand its structure. Do not write any code yet.
→ Announce: "代码库已阅读完毕。等待 `design-{id}.md` 定稿后再开始实现。"

---

## Path A — Adapt

The reference codebase lives at `srcPath` from `.oma/codebase/config.json`.
Codex has full access to read any file in that directory.

### Step 1 — Read the design, then read the code

From `design-{id}.md`, extract every component that needs implementing:
architecture layers, loss function, optimizer config, data pipeline changes, ablation variables.

For each design component, locate the corresponding file(s) in the reference codebase:
- Read the relevant files directly — Codex can open any file in `srcPath`
- Understand the existing implementation: class structure, forward pass, data flow
- Identify exactly what needs to change to match the design

### Step 2 — Implement directly

Make the changes. Work component by component in this order:
1. Model / architecture changes
2. Loss function changes
3. Data pipeline changes
4. Optimizer / scheduler changes
5. Entry point wiring (connect new components, expose new hyperparams in config)

For every change, mark it clearly:
```python
# [OMA] {design-id} — {brief description of what changed and why}
```

Rules:
- Change only what the design requires. Leave unrelated code untouched.
- If an existing module is fundamentally incompatible with the design, rewrite only that module.
- All ablation variables from `design-{id}.md` must be wired to the config file, not hardcoded.
- Preserve existing logging, checkpointing, and data loading unless the design explicitly changes them.

### Step 3 — Smoke test

Run 10 training steps with a minimal config to confirm the changes work end-to-end.

If the smoke test fails: fix and re-run. Do not proceed to the checklist with broken code.

### Step 4 — Write impl-checklist.md

```markdown
# Implementation Checklist (Path A — Adapt)
_Design: {design-id}_
_Reference repo: {srcPath}_
_Date: {date}_

## Design Components Implemented
- [ ] {Component 1}: {file modified} — {what changed}
- [ ] {Component 2}: ...
- [ ] All ablation variables wired to config (not hardcoded)
- [ ] New hyperparams exposed in entry point / config schema

## Reproducibility
- [ ] Seed setting confirmed: {framework}={seed}
- [ ] Non-deterministic ops documented: {list or "none"}

## Smoke Test
- [ ] 10-step run completed without crash
- [ ] Loss: {start value} → {end value} (no NaN, no inf)
- [ ] Checkpoint saved
```

Mark each item `✓` only after verifying it — not after writing the code.

---

## Path B — Scratch

> **注（2026-09-24）**：Isaac Gym 类机器人 RL 项目**恒走 Path A**（代码基座改造）；本 Path 的从零脚手架（dataset/loader/loss 结构）为通用 ML 遗产，不适用于 GPU 仿真项目——保留仅供参考。

### Step 1 — Scaffold

Create the project structure matching the framework in `requirements.md`:

```
train.py              ← entry point: --config <path>
configs/
  base.json           ← all hyperparams from design-{id}.md
src/
  model.py            ← architecture
  data.py             ← dataset + loader
  loss.py             ← loss function(s)
  metrics.py          ← metric computation
  utils/
    seed.py           ← reproducibility: set_seed(n) touches all frameworks
```

Adapt to the project's framework (Lightning, HF Trainer, JAX/Flax, etc.) if specified.

### Step 2 — Build in order

Build strictly in dependency order:

**Data pipeline first** — everything else depends on knowing the input shape.
- Dataset class, train/val/test splits matching `requirements.md`
- Preprocessing and normalization from design spec
- Deterministic worker seeds
- Verify: load 2 batches, print shapes, confirm no NaN

**Model second**
- Implement architecture layer-by-layer exactly as in `design-{id}.md`
- Log parameter count on init
- Verify: forward pass with random input → expected output shape

**Loss third**
- Primary loss from design; auxiliary losses if any
- Verify: sanity check — known input produces expected loss range (not NaN)

**Training loop fourth**
- Optimizer and LR scheduler from design defaults
- Gradient clipping if specified
- Mixed precision if compute budget requires
- Checkpoint: every N steps + best val metric (configurable)
- Metric logging: each step writes `{"step": N, "loss": X, "metric": Y}` to stdout

**Entry point last**
- `train.py --config configs/base.json`
- On startup: write resolved config to `.oma/experiments/{exp-id}/config.json`
- On completion: write `.oma/experiments/{exp-id}/results.json`

### Step 3 — Smoke test

10 training steps with a tiny batch. Loss must move (even slightly). No crash, no NaN.

### Step 4 — Write impl-checklist.md

```markdown
# Implementation Checklist (Path B — Scratch)
_Design: {design-id}_
_Date: {date}_

## Data Pipeline
- [ ] Dataset loads without error
- [ ] Shapes: input={shape}, output={shape}
- [ ] No NaN in first 2 batches
- [ ] Splits: train={N}, val={N}, test={N}

## Model
- [ ] Architecture matches design-{id}.md layer by layer
- [ ] Parameter count: {N}
- [ ] Forward pass output shape: {shape}

## Loss
- [ ] {Loss name} implemented
- [ ] Sanity check passed

## Training Loop
- [ ] Optimizer: {name}, lr={value}
- [ ] LR scheduler: {type}
- [ ] Gradient clipping: {value or none}
- [ ] Checkpoint: every {N} steps + best val metric
- [ ] Metric logging to stdout per step

## Entry Point
- [ ] `train.py --config <path>` runs without error
- [ ] Config written to experiment dir on startup
- [ ] `results.json` written on completion

## Reproducibility
- [ ] Seeds: {framework}={seed}
- [ ] Non-deterministic ops: {list or none}

## Ablation Readiness
- [ ] All ablation variables from design are config-controlled (not hardcoded): {list}

## Smoke Test
- [ ] 10-step run: no crash, no NaN
- [ ] Loss: {start} → {end}
- [ ] Checkpoint saved
- [ ] `results.json` written
```

---

## GitHub Push — Mandatory Final Step

Training on Gradmotion pulls code exclusively from GitHub (`codeType=2`). The push is part of `$implement`, not `$train`.

**Do this after smoke test passes and checklist is complete:**

```bash
git add -A
git commit -m "[OMA] {design-id} — implement {brief summary of changes}"
git push origin {branch}
```

Then record the push in `.oma/impl/github.json`:

```json
{
  "repoUrl": "https://github.com/{org}/{repo}.git",
  "branch": "{branch}",
  "commitHash": "{full SHA from git rev-parse HEAD}",
  "pushedAt": "{ISO timestamp}",
  "designId": "{design-id}"
}
```

If the repo does not yet exist or the remote is not configured, ask the user to provide the GitHub repo URL before pushing.

If the repo is **private**: remind the user that Gradmotion needs their GitHub credentials configured on the platform (Web UI → 个人设置 → Git 信息) before `$train` can pull the code.

---

## Completion Signal

Complete when:
1. `impl-checklist.md` exists with every item `✓`
2. Smoke test passed（暂缓期间：首个 GM 云端任务代行，见头部声明）
3. All ablation variables are config-controlled
4. Code pushed to GitHub and `.oma/impl/github.json` written

Report:
```
Implementation complete.
  Path: {A — Adapt / B — Scratch}
  Smoke test: {start loss} → {end loss} over 10 steps
  Ablation variables: {list from design}
  GitHub: {repoUrl} @ {branch} ({short commit hash})
  Training admission: {bounded probe / same-contract formal proof / blocked; actual evidence}
```

---

## Bug-Fix Re-entry (from $train failure)

If `$train` returns with a code error, re-enter `$implement` here:

1. Read the error from `.oma/experiments/{exp-id}/error.log`
2. Locate the failing file and fix the bug
3. Re-run the smoke test (10 steps) to confirm the fix
4. Push the fix:
   ```bash
   git add -A
   git commit -m "[OMA] fix: {one-line description of bug}"
   git push origin {branch}
   ```
5. Update `.oma/impl/github.json` with the new `commitHash` and `pushedAt`
6. Return to `$train` — it will use `flux task copy` to launch a fresh run from the same config
