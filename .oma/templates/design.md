# Design — {PROJECT_NAME}
_ID: design-{YYYYMMDD}-{SEQ}_
_Created: {DATE}_
_Status: DRAFT — becomes LOCKED after $design Phase 4_
_Knowledge source: {knowledge.md paper reference}_

<!--
  USAGE:
  - Written by $design Phase 4.
  - Read by $implement, $tune.
  - LOCKED after $design completes.
  - Each field should record its provenance (Stream A/B/C + source).
-->

## Idea Pool Summary
<!-- Which ideas were generated, evaluated, and why this one was selected -->

| ID | Title | Stream | Priority | Decision |
|----|-------|--------|----------|---------|
| A1 | {title} | A | {score} | {selected / skipped: reason} |
| B1 | {title} | B | {score} | {selected / skipped: reason} |
| C1 | {title} | C | {score} | {selected / skipped: reason} |

**Selected**: {Idea ID} — {one-sentence justification}
**Human input**: {what the human selected/modified/hinted}

---

## Primary Design

### Goal alignment
- Goal version / source: {requirements.md 当前版本与用户确认出处}
- Deliverable behavior and control split: {输入 → 实际行为；学习 / 参考等分工}
- Covered goal IDs / missing scope: {引用需求覆盖表，不把阶段实现称为总体交付}
- First behavior review: {已有方向认可证据 / 首次展示计划及局限}
- Revision impact (if any): {理解纠正 / 细节补充 / 目标变更；适用实验与保留旧判定}

### Provenance
- **Core idea**: {Idea ID} [{Stream A/B/C}]
- **Source**: {paper title / limitation text / cross-domain analogy}
- **Human modification**: {description or "none"}

### Problem Statement (from current requirements.md)
- Task: {用户当前交付目标；论文问题仅作方法依据}
- Primary metric / quality acceptance: {定义、来源与视觉 / 人工范围}
- Applicable baseline: {实际结果与协议；论文指标不自动成为必须超越的目标}

### Architecture

- Training/deployment contract: {本轮 config.deployment 或原定义引用；不另建审查文件}
- Actual deployment sources / critical unknowns: {从用户提供代码反向核对 actor 实际可得性、预处理 / 时序、动作 / PD / 保护 / 状态机；未知阻塞正式部署训练}
- Admission evidence / next validation: {有界 probe 的真实上限与失败分支，或同契约实际新权重的部署闭环证据；接口通过不代替行为通过}
<!-- Concrete layer-by-layer specification. No vague descriptions. -->

**Input preprocessing**
- {step 1: e.g., "normalize features to zero mean, unit variance using training statistics"}
- {step 2}

**Model structure**
```
Layer 1: {type} — input: {shape}, output: {shape}, params: ~{N}
Layer 2: {type} — input: {shape}, output: {shape}, params: ~{N}
...
Output head: {type} — {shape} → {shape}
Total parameters: ~{N}
```

**Design rationale**: {why this architecture — link to provenance}

### Loss Function
- Primary loss: {name} — formula: {LaTeX or plain text}
- Auxiliary losses (if any): {name}, weight: {λ}
- Known failure modes of this loss: {e.g., "sensitive to class imbalance"}

### Optimization
| Hyperparameter | Value | Source |
|---------------|-------|--------|
| Optimizer | {e.g. AdamW} | {from paper / proposed} |
| Learning rate | {value} | {from paper / proposed} |
| LR schedule | {type} | {from paper / proposed} |
| Warmup | {steps or ratio} | {proposed} |
| Batch size | {value} | {from paper / proposed} |
| Gradient clip | {value or "none"} | {proposed} |
| Weight decay | {value} | {proposed} |
| Dropout | {value} | {proposed} |
| Epochs | {value} | {proposed} |

### Reproducibility
- Global seed policy: Python={seed}, NumPy={seed}, {framework}={seed}
- Non-deterministic ops: {list or "none identified"}
- Data loading: worker seeds set to base_seed + worker_id

---

## Ablation Plan
<!-- Each row = one $tune experiment variable -->

| Variable | Candidates | Hypothesis | Priority | Linked to design decision |
|---------|-----------|-----------|---------|--------------------------|
| {e.g. attention_type} | [none, dot-product, additive] | {expected outcome} | high | Layer 2 design |
| {e.g. loss_weight_λ} | [0.1, 0.5, 1.0] | {expected outcome} | medium | Loss function |
| {e.g. warmup_ratio} | [0.0, 0.05, 0.10] | {expected outcome} | low | Optimization |

Sweep strategy for $tune: {sequential ablation / grid / directed}
Estimated configurations: {N} × {seeds} = {total runs}

---

## Alternative Designs (Fallbacks)

### Alternative 1 — {Idea ID}: {Title}
_Use if primary design fails to beat baseline after {N} training runs_

- Core difference from primary: {one sentence}
- Provenance: Stream {X} — {source}
- Expected: {metric estimate}
- Key risk: {what could go wrong}

### Alternative 2 — {Idea ID}: {Title} (optional)
_Use if Alternative 1 also underperforms_

- Core difference: {one sentence}
- Provenance: Stream {X} — {source}

---

## Design Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| {e.g. "overfitting on small dataset"} | medium | {e.g. "strong dropout + early stopping"} |
| {e.g. "training instability at high lr"} | low | {e.g. "warmup + gradient clipping"} |
| {e.g. "cross-domain transfer fails"} | medium | {fallback to Alternative 1} |

---

## Dead Ends Confirmed (do not revisit)
<!-- From knowledge.md Dead End Seeds + memory.md, relevant to this design space -->

| Direction | Source | Reason |
|-----------|--------|--------|
| {direction} | {knowledge.md / memory.md} | {why not viable} |

---
_Provenance: idea pool from $design (Stream A: {N} ideas, Stream B: {N} ideas, Stream C: {N} ideas)_
_Locked by: $design on {date}_
_Next step: $implement_
