# Skill: $requirement

**Purpose**: 将用户意图转成可观察的交付行为、任务边界与验收证据。论文提供方法和外部事实；项目目标由用户指令确定。支持首次澄清及循环中的理解纠正、细节补充和目标变更。

**Gate in**: None. This is the entry point of the OMA lifecycle.
**Standalone entry**: Always allowed. 从已有上下文提取交付行为、目标平台和验收方式，只问影响当前路线的缺项；无论文也可建立任务目标。
**Gate out**: Creates two locked artifacts:
- `.oma/knowledge.md` — external literature knowledge (read-only after this skill)
- `.oma/requirements.md` — 版本化交付目标、确认来源、覆盖表与成功标准（gate chain source）

---

## Overview

先读 [目标对齐规约](goal-alignment.md)。所有模式均采用：正式投入前确认交付含义、首次行为可见时确认方向、中途修订时确认影响范围。已有明确指令即可作为确认依据，不重复索要同一授权。

- **首次需求**：先读上下文和项目资产；有论文走 Phase 1–2，无论文从项目实际资料形成 knowledge.md，缺外部知识明确写未知，再走 Phase 3–5。
- **已锁定目标的补充 / 纠正**：直接走 Phase 6，保留旧版本和实验，不重新抽论文或启动完整问卷。
- **数值来源**：区分用户要求、项目实测、论文事实和工程设置；无依据写未知，不从通用惯例补硬件参数、费用或用户阈值。

---

## Phase 1 — Paper Ingestion

### 1a. Check for pre-extracted content

First, check if `oma extract` was already run:
- If `.oma/paper/raw-sections.json` exists → read it and skip to Phase 2.
- If `.oma/paper/knowledge-draft.md` exists → read it and skip to Phase 3.

### 1b. Request paper if not pre-extracted

If no pre-extracted content:

```
I need a benchmark paper to anchor the requirements. Please provide one of:

  Option A: Run `oma extract --paper path/to/paper.pdf` before this session,
            then restart the $requirement workflow.

  Option B: Paste the key sections directly into this session:
            - Abstract (full text)
            - Related Work or Background (full text)
            - Experiments / Evaluation Setup (full text)
            - Limitations or Conclusion (full text)
```

Wait for the user to provide content. Do not proceed to Phase 2 without it.

### 1c. Identify and extract sections

From the paper text, extract the following sections. Use section header matching (case-insensitive, allow numbering like "2. Related Work" or "§3 Experiments"):

| Section | Purpose | Priority |
|---------|---------|----------|
| Abstract | Problem definition in one paragraph | high |
| Introduction | Gap being addressed, why existing methods fail | high |
| Related Work / Background | Algorithm family landscape | high |
| Method / Approach | Innovation claims | medium |
| Experiments / Evaluation | Dataset, metrics, baselines, reported numbers | **critical** |
| Limitations / Conclusion | Known failure modes, open problems | high |

If a section cannot be found, mark it as `[not found]` and note which information will need to come from the user interview instead.

---

## Phase 2 — Synthesize knowledge.md

Using the extracted sections, produce `.oma/knowledge.md` with the following schema. Be specific — extract actual numbers, dataset names, metric formulas from the paper text. Do not paraphrase vaguely.

```markdown
# Knowledge Base
_Source: {full paper title} — {authors} ({year})_
_Venue: {conference/journal or "preprint"}_
_Extracted: {date}_
_Status: LOCKED — do not modify after $requirement completes_

## Problem Formulation
- Task: {one-sentence task description from abstract}
- Input: {data format, modality}
- Output: {prediction format}
- Core challenge the paper addresses: {one sentence}

## Algorithm Landscape
| Family | Representative Papers | Strength | Weakness |
|--------|----------------------|---------|---------|
| {e.g. gradient boosted trees} | {paper names} | {strength} | {weakness} |

Synthesis: {1–2 sentences on which families are dominant and why}

## Innovation Claims (what this paper does differently)
- {Claim 1: specific and falsifiable}
- {Claim 2}
- {Claim 3}

Note: These become candidate directions for $design. They represent the paper's
      proposed solution — not necessarily what we will implement.

## Evaluation Protocol (from Experiments section)
- Dataset: {name, size, train/test split if stated}
- Primary metric: {name + formula if non-standard}
- Secondary metrics: {list}
- Baselines compared against: {list with their reported scores}
- Best result in paper: {metric name}: {value} (on {dataset}, {split})
- Evaluation protocol details: {e.g. 5-fold CV, single test set, specific seeds}

## Known Limitations (from paper)
| Limitation | Source location | Implication for our work |
|-----------|----------------|--------------------------|
| {limitation text} | {section} | {what this means for our constraints} |

## Reproduction Constraints
- Compute reported: {e.g. "8×A100 GPUs, 3 days" or "not stated"}
- Data availability: {public / proprietary / partial}
- Code released: {yes + URL / no / partial}
- Special infrastructure: {e.g. "requires FPGA" or "none"}

## Dead End Seeds
<!-- These will be transferred to memory.md by $consolidate after experiments begin -->
Directions the paper explicitly found not to work:
- {direction}: {why it failed per paper}
```

**Quality check before proceeding:**
- [ ] Evaluation Protocol 中已报告数值逐项有论文出处；未报告时明确写未报告，不编数值以补齐模板。
- [ ] Known Limitations has at least one entry (if the paper has none, note "paper reports no limitations").
- [ ] Algorithm Landscape 只包含已查证的方法；未提供足够材料时写明范围，不虚构方法以满足条数。

若材料与记录不一致，回查来源；原文确实没有的信息保留缺项，不将论文基线自动转成项目成功标准。

---

## Phase 3 — Draft the delivery contract

使用 `templates/requirements.md`（安装后为 `.oma/templates/requirements.md`），结合 [目标对齐规约](goal-alignment.md) 填写；优先复用已存在的用户指令与项目数据。

1. 用用户能判断的行为复述最终交付：输入 / 控制方式、期望行为、阶段 / 工况和持续运行需求。
2. 标明算法 / 控制分工、交付环境、质量要求、可接受不足及非目标。只展开会影响路线的选择。
3. 建立“目标 → 协议 → 实际指标 / 视觉证据”的覆盖表；软件接入、短演示与总体能力分别标注。
4. 每个关键结论保留来源。论文指标是外部参考，不能代替用户目标或自动成为部署阈值；暂不能量化的形态要求保留人工 / 视频验收。

## Phase 3b — Robot platform evidence

机器人项目从实际配置 / URDF / 资产与已确认指令提取 robot、DoF、仿真器、控制周期、观测 / 动作、执行器及部署约束，逐项标来源。不能把论文机器人、典型 PD、默认频率或惯例随机化范围当作本项目事实。

缺失字段标未知及其影响：阻碍当前实现 / 训练的字段优先核查；只有用户能决定且影响路线的选择才询问。与当前仿真阶段无关的部署细节可保留待验，不为填满模板阻塞独立工作。

## Phase 4 — Targeted alignment

先向用户给出简短的行为复述和已知边界；只问会改变路线、验收或授权范围的未决选择，并给推荐及理由。

- 论文复现 / 同架构功能适配 / 方法探索，从用户已有表达判断，不能只按数据集是否相同推断；不清楚且影响方案时才问。
- 机器人行为任务尤其检查固定轨迹与指令闭环、纯策略与允许参考控制、短段演示与持续能力是否有歧义。无需给每个任务机械套用全部问题。
- 普通实现细节自主决定，不向用户派发长规格表。未知数值先查项目数据，不能让用户随意给阈值替代事实。
- 当前指令已明确的内容直接记录来源；没有新的未决选择时直接继续。未答复不是同意，也不能把工程提案标为用户已确认。

影响路线的未决选择解决前，只推进不依赖该选择、已有授权的工作。方向验收安排在首次有效行为可见时，不要求在没有行为产物时先获得视觉通过。

## Phase 5 — Lock the current version

### 5a. Knowledge

保留 knowledge.md 的外部事实及缺项，设为 `Status: LOCKED`；无论文时明确“项目资料 / 无论文”，不生成虚构论文结论。后续意图修订不改论文知识。

### 5b. Requirements

- 保存当前需求版本、交付复述、用户来源、未决问题、覆盖表和首次方向验收安排。
- `LOCKED` 表示当前有效依据；锁定前解决影响本阶段正式投入的路线歧义。非阻塞未知保留来源 / 后续核查点，不能靠删标签伪造已确认。
- 保留事实 / 提案 / 用户要求的来源标签。阈值按项目目标与实际依据制定，不要求高于论文结果，也不要求所有自然度目标立即数值化。
- 将版本及交付锚同步到任务卡 §0，覆盖与判定写入 §2 / §6；不另造平行目标账本。任务卡尚未创建时由后续 design 填写。
- 既有需求已 LOCKED 但缺新增字段时，增量补齐可确定内容，只问影响本次决定的真实缺项；不机械重启旧任务。

### 5c. Memory and checks

memory.md 不存在时按现有模板初始化，真实论文 Dead Ends 标明出处；无资料则留空。锁定前核查：

- 交付行为、控制边界与确认来源是否一致，路线关键歧义是否已解决。
- 覆盖表是否明确当前轮范围与未覆盖项，首次方向验收是否有计划 / 既有来源。
- 每个数值是否有实际出处或明确工程设置标签；平台参数与本次实现是否相符。
- 目标与部署授权是否分开；未解决的后续字段是否仍显式可见。

完成时简短报告：当前需求版本、交付行为、有效确认来源、未覆盖项 / 下一方向验收点。随后按既有授权进入设计，不为相同目标再次索要确认。

## Phase 6 — Incremental correction or revision

按 [目标对齐规约](goal-alignment.md) 分清理解纠正、细节补充、目标变更；可按条目分类，不把 agent 误读全部记成用户改需求。

1. 读取当前版本、用户原意与实验范围；保留修订前受影响文字，登记来源、分类、差量、适用起点和设计 / 实验 / 协议影响。
2. 更新 requirements.md 与任务卡中的有效目标及覆盖；受影响设计 / 未来训练计划重新对齐。已有明确修订指令即为依据，不重问原访谈。
3. 原登记实验按原冻结门回填，另外标记对新目标的适用性或协议缺陷；不追溯造成功，不删除旧证据，不手改 decision_log。
4. 当前在跑任务的继续 / 停止按实际授权处理；目标修订本身不等于取消训练。方向认可、版本接受、总体收束和真机批准分别记录。

## Standalone entry

同样使用最小交付契约和增量路径；不阻塞于论文缺失、不从零重问已有目标。Standalone 免前置文档门不等于可以假定用户意图、行为验收或执行授权。
