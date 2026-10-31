# Project Model Culi — Admin Transfer Handover Package

**Document:** `/root/culi/PROJECT_HANDOVER.md`
**Purpose:** Complete handover for the incoming Project Strategist (PS).
**Transfer trigger:** User command "đổi admin".
**Scope:** Research and documentation only. No implementation authorized.

---

## 1. Project Objectives and Approved Decisions

### 1.1 Established Objectives

Project Model Culi researches and develops **one unified neural model** capable of:

1. Vietnamese and English language understanding and generation.
2. Rust programming comprehension and code generation.
3. Dioxus framework comprehension and application.
4. General reasoning and problem-solving.
5. Planning and autonomous task execution.

### 1.1 Strategic Direction (D-0006)

Project Model Culi now aims to discover and develop a new unified neural architecture for strong general intelligence and exceptional resource efficiency on low-end hardware. Existing technologies are references, not design boundaries. Performance is a foundational design objective, not a later optimization.

Six foundational principles:

1. New Architecture: Discover and develop potentially new neural architectures rather than assuming existing architecture families define the available solutions.
2. Unified Intelligence: One neural model owns all core intelligence, including knowledge application, understanding, reasoning, planning, decision-making, and tool selection.
3. High Intelligence: Support Vietnamese, English, Rust, Dioxus, general reasoning, problem-solving, and autonomous task execution.
4. Maximum Efficiency: Maximize useful intelligence and task capability per unit of computational resources.
5. High Performance: Prioritize compact model size, fast response, high throughput, low latency, and efficient RAM and CPU utilization.
6. Break Limits: Challenge existing technical limitations and explore fundamentally new solutions. Scientific and experimental evidence must validate every claim.

Agent and tool philosophy: Culi must not recreate an existing agent framework unnecessarily. The neural model should possess the intelligence to understand, select, control, and evaluate external agent frameworks and tools. External frameworks, compilers, terminals, file systems, and execution tools may provide capabilities and perform operations, but they must not replace or supply Culi's core intelligence.

Core requirements:

- **Unified Model Rule:** All core intelligence resides in one unified neural model. An external execution mechanism may execute commands or interact with the OS, but must not introduce another model responsible for core intelligence.
- **Hardware Constraint:** Ubuntu VPS, 4 GB RAM, conventional CPU, no dedicated GPU. The 4 GB target applies to both training and inference. Exceptions require explicit project-owner approval.
- **Architecture Independence:** Transformer, RNN, SSM, MoE, recursive networks, and other approaches are research subjects, not predetermined solutions.
- **Documentation Language:** Professional technical English. Vietnamese only in communication with the project owner.
- **Workspace Rule:** All project-owned documentation, research records, source code, and experiment records remain under `/root/culi`.

### 1.2 Approved Decisions (see `docs/DECISIONS.md`)

| ID | Decision |
|---|---|
| D-0001 | Initialize Project Model Culi under `/root/culi`. |
| D-0002 | Adopt approved directory structure. |
| D-0003 | Record unified-model objective and 4 GB CPU-only constraint as project constants. |
| D-0004 | Treat existing architectures as research subjects, not predetermined solutions. |
| D-0005 | Operational definition of 4 GB RAM: max 4 GB per training process and per inference process, measured separately, excluding swap, based on peak process memory. |
| D-0006 | Strategic direction (D-0006): discover and develop a new unified neural architecture for strong general intelligence and exceptional resource efficiency on low-end hardware; existing technologies are references, not design boundaries; performance is a foundational design objective; the model natively understands, selects, controls, and evaluates external tools and agent frameworks without unnecessarily recreating an existing agent framework. |
| D-0007 | Minimal standard-library development foundation (D-0007): repository validation and peak-RSS measurement utilities under `src/` and `tests/`. |
| D-0008 | Architecture-independent intelligence evaluation system (D-0008): structured datasets, automatic and manual scoring, JSON export, model-version comparison, and mock adapter. |
| D-0009 | Experimental feedforward neural prototype (D-0009): training, saving, loading, inference adapter, prototype XOR dataset, and evaluation integration. |
| D-0010 | Controlled comparison experiment (D-0010): feedforward vs recurrent prototypes on XOR across seeds 1-3. |
| D-0011 | Sequential memory and reasoning experiment (D-0011): synthetic tasks, multiple models, deterministic seeds, familiar and longer lengths. |
| D-0012 | Memory organization experiment (D-0012): gated-memory and hierarchical-state candidates, multiple tasks, deterministic seeds, familiar and longer lengths. |

### 1.3 Operational 4 GB Definition (D-0005)

- Max 4 GB RAM per training process.
- Max 4 GB RAM per inference process.
- Training and inference peak memory measured separately.
- Swap excluded.
- Peak process memory measured, not average.
- Includes parameters, gradients, optimizer states, activations, runtime buffers, and attributable framework overhead.

---

## 2. Current Status and Completed Work

### 2.1 Status

- Phase: Research Foundation complete; research planning pending project-owner approval.
- Architecture: No architecture selected. No implementation exists.
- Git: Not initialized. No packages installed. No data or weights downloaded. No training performed.
- Reserved directories (`research/`, `experiments/`, `src/`, `tests/`, `models/`) are empty.
- Experiment log: no experiments recorded.
- Decision log: D-0001 through D-0013 approved; remaining decisions TO BE DEFINED.

### 2.2 Completed Work

- Project workspace and documentation scaffold created.
- Environment verification audit (PASS).
- Architecture Research Foundation report produced.
- Technical correction review produced (memory accounting, CPU BF16, inference complexity, bibliography, 4 GB operational definition).
- Conceptual review of an external implementation proposal (rejected as implementation plan; elements retained as unapproved hypotheses).
- Operational 4 GB RAM constraint standardized in documentation (D-0005).
- New strategic direction standardized in documentation (D-0006).
- Minimal standard-library development foundation implemented (D-0007).
- Architecture-independent intelligence evaluation system implemented (D-0008).
- Experimental feedforward neural prototype implemented and tested (D-0009).
- Controlled feedforward vs recurrent comparison executed (D-0010).
- Sequential memory and reasoning experiment executed (D-0011).
- Memory organization experiment executed (D-0012).
- Selective memory and long-range retention experiment executed (D-0013).

---

## 3. Research Findings and Unresolved Questions

### 3.1 Key Findings

- Memory feasibility is plausible for small models (tens of millions of parameters) with careful budgeting, checkpointing, and quantization; capability feasibility at this scale is unverified for all families.
- 12 bytes/parameter applies only to pure low-precision training without FP32 master weights. Standard mixed precision with master weights yields 16N or 18N.
- Activation memory dominates and depends on implementation (coefficient c = 12-20). Gradient checkpointing savings are strategy-dependent, not a fixed 1/sqrt(L).
- CPU BF16 support is hardware- and backend-dependent; conventional VPS CPUs may lack efficient BF16 execution.
- Transformer inference must distinguish prefill O(T^2) from per-token decoding O(T) attention plus O(D^2) FFN; KV cache grows with context length.
- Sparse/conditional computation reduces active FLOPs, not necessarily RAM, because total parameters remain stored.
- Byte-level processing is likely memory-negative (small embedding savings vs. 4-6x longer sequences and activations).
- Capability claims in literature (reasoning, planning, agent/tool-use) derive from training methods, data, scale, and system orchestration, not architecture alone.
- Candidate families retained as hypotheses only: modern Transformer variants, selective SSM (Mamba/Mamba-2), and hybrid SSM + sparse/local attention. No family is selected or endorsed.

### 3.2 Unresolved Questions (TO BE DEFINED)

- Neural architecture family; layer/block design.
- Parameter, memory, and context-length budgets.
- Tokenizer/vocabulary design or byte-level approach.
- Training data sources, licenses, curation, and synthetic data policy.
- Training methodology and baseline models.
- Evaluation benchmarks, metrics, and acceptance thresholds.
- Inference latency/throughput targets.
- Quantization/compression/deployment strategy.
- Representation of planning and autonomous task execution.
- Safety/alignment, privacy/governance, license, contribution policy, team roles.
- Research tooling, dependency management, runtime versions, reproducibility, storage, artifact retention.

### 3.3 Unverified Hypotheses

- Any architecture at 4 GB training scale can deliver all five capabilities.
- SSM state capacity suffices for Rust/Dioxus code recall.
- Retrieval augmentation improves capability metrics within memory budget.
- Tool-augmented agent loops achieve task success at this scale.
- Byte-level processing is net beneficial.
- Sparse/conditional computation fits within 4 GB given total parameter storage.
- Hybrid SSM + attention captures the best of both families.
- Offline distillation transfers reasoning/code capability to a small model.

---

## 4. Current Tasks and Recommended Next Steps

### 4.1 Current Tasks

- Research planning pending project-owner approval.
- Literature review priorities not yet selected.
- 4 GB operational definition has open confirmation items (physical vs. available RAM, coexistence of training/inference, headroom, allocator caches).
- Evaluation principles and benchmarks not yet defined.
- Milestones 6, 7, and 8 have been implemented and verified; documentation reconciled.
- Git repository initialized; GitHub remote configured; baseline commit created.

### 4.2 Recommended Next Steps (Owner Approval Required)

1. Approve research scope and initial research questions.
2. Select literature review priorities across architecture families.
3. Define evaluation principles before architecture comparison.
4. Confirm operational 4 GB ambiguities.
5. Establish parameter and memory budget targets.
6. Define tokenizer/vocabulary or byte-level hypothesis.
7. Define data sources, licenses, and curation policy.
8. Define evaluation benchmarks, metrics, and acceptance thresholds.
9. Design controlled experiments for candidate architectures.
10. Record all decisions in `docs/DECISIONS.md` and experiments in `docs/EXPERIMENT_LOG.md`.

---

## 5. Rules and Restrictions for Operating 108

108 is the Cline/AI agent instance. The following rules are mandatory:

1. Operate only within `/root/culi`. Do not create, modify, or delete files outside it.
2. Do not implement or train any model.
3. Do not install packages or extensions.
4. Do not download datasets or model weights.
5. Do not call paid APIs.
6. Do not modify VPS configurations.
7. Do not initialize Git without explicit approval.
8. Treat architectures as research subjects, not predetermined solutions. Do not select a final architecture without approval and evidence.
9. Do not claim any architecture meets the 4 GB constraint without a defensible memory model or experimental evidence.
10. Do not treat hypotheses as validated conclusions. Distinguish established objectives, unresolved questions, hypotheses, and validated conclusions.
11. All documentation and technical artifacts in professional technical English. Vietnamese only in communication with the project owner.
12. Do not redefine established project objectives.
13. Record decisions in `docs/DECISIONS.md` and experiments in `docs/EXPERIMENT_LOG.md`.
14. Require explicit project-owner approval for implementation, dependency installation, data acquisition, training, Git initialization, hardware exceptions, and scope changes.
15. After any modification, report the complete affected directory tree, all created/modified files, document summaries, confirmation of no out-of-workspace files, and any unresolved issues.

---

## 6. Instructions for the New PS to Manage, Supervise, and Assign Tasks to 108

- Treat 108 as a research/execution assistant, not a decision-maker.
- Assign only tasks within approved scope and rules. Reject or escalate scope creep.
- Provide each task with explicit scope, inputs, expected outputs, and constraints.
- Require 108 to read `.clinerules/project-rules.md` and relevant `docs/` before acting.
- Require 108 to report affected files, summaries, and confirmations after each task.
- Verify outputs against project documents and constraints. Do not accept unverified claims.
- Ensure 108 records decisions in `docs/DECISIONS.md` and experiments in `docs/EXPERIMENT_LOG.md`.
- Escalate uncertainties and approval-gated items to the project owner. 108 must not make strategic decisions on behalf of the owner.
- For research tasks, require evidence/hypothesis/validated-conclusion distinctions.
- For documentation tasks, enforce professional technical English and workspace-only changes.

---

## 7. Exact Startup Prompt for 108 (New Session)

Use the following prompt verbatim to initialize 108:

```
You are Cline instance 108 assigned to Project Model Culi. Operate only within /root/culi. Before any action, read and obey .clinerules/project-rules.md and all files under docs/. This is a research-and-documentation phase unless explicitly authorized otherwise. You must not implement or train models, install packages, download datasets or weights, call paid APIs, modify VPS configuration, or initialize Git. Distinguish established objectives, unresolved questions, hypotheses, and validated conclusions. Do not redefine project objectives or make strategic decisions for the project owner. First action: verify the environment and summarize current project state. Then wait for further instructions. Acknowledge these constraints before proceeding.
```

---

## 8. Uncertainties and Items Requiring Project-Owner Approval

- Operational 4 GB ambiguities (physical vs. available RAM, coexistence, headroom, allocator caches).
- Research scope and literature priorities.
- Parameter, memory, and context-length budgets.
- Tokenizer/vocabulary vs. byte-level approach.
- Data sources, licenses, curation, and synthetic data policy.
- Evaluation benchmarks, metrics, and acceptance thresholds.
- Inference latency/throughput targets.
- Quantization/compression/deployment strategy.
- Planning/agent representation.
- Safety/alignment, privacy/governance, license, contribution policy, team roles.
- Architecture selection (no architecture selected or endorsed).
- Any exception to the 4 GB/CPU-only constraint.

---

## 9. Mandatory Documentation Synchronization

This rule applies to every current and future implementation task.

1. Before implementation, read the relevant project documentation.
2. During implementation, maintain documentation alongside code changes.
3. After implementation, update all affected documentation to accurately reflect the actual code, behavior, configuration, commands, and test results.
4. Update `PROJECT_MASTER.md`, `ARCHITECTURE.md`, `README.md`, `DECISIONS.md`, and other relevant documents whenever their contents are affected. Do not modify unrelated documents unnecessarily.
5. Record new technical decisions, changes in constraints, and deviations from approved plans.
6. Never document planned or intended behavior as if it were already implemented.
7. If implementation fails or remains incomplete, document the actual status and blockers.
8. Before finishing, verify that source code, tests, configuration, and documentation are consistent.

An implementation task is not complete until both the implementation and its corresponding documentation have been updated and checked.

## 10. Key References

- `README.md` — project overview and boundaries.
- `.clinerules/project-rules.md` — project rules.
- `docs/PROJECT_MASTER.md` — canonical charter.
- `docs/ARCHITECTURE.md` — architecture status and research subjects.
- `docs/RESEARCH_ROADMAP.md` — research phases and milestones.
- `docs/TECHNICAL_CONSTRAINTS.md` — technical and hardware constraints.
- `docs/AGENT_RULES.md` — agent operating rules.
- `docs/DECISIONS.md` — decision log.
- `docs/EXPERIMENT_LOG.md` — experiment index and conventions.
- `research/`, `experiments/`, `src/`, `tests/`, `models/` — reserved for future work.
