# Project Master

## 1. Purpose

This document is the canonical charter for Project Model Culi. It records established objectives, constraints, status, and unresolved technical questions.

## 1.1 Strategic Direction (D-0006)

Project Model Culi now aims to discover and develop a new unified neural architecture for strong general intelligence and exceptional resource efficiency on low-end hardware. Existing technologies are references, not design boundaries. Performance is a foundational design objective, not a later optimization.

Six foundational principles:

1. New Architecture: Discover and develop potentially new neural architectures rather than assuming existing architecture families define the available solutions.
2. Unified Intelligence: One neural model owns all core intelligence, including knowledge application, understanding, reasoning, planning, decision-making, and tool selection.
3. High Intelligence: Support Vietnamese, English, Rust, Dioxus, general reasoning, problem-solving, and autonomous task execution.
4. Maximum Efficiency: Maximize useful intelligence and task capability per unit of computational resources.
5. High Performance: Prioritize compact model size, fast response, high throughput, low latency, and efficient RAM and CPU utilization.
6. Break Limits: Challenge existing technical limitations and explore fundamentally new solutions. Scientific and experimental evidence must validate every claim.

Agent and tool philosophy: Culi must not recreate an existing agent framework unnecessarily. The neural model should possess the intelligence to understand, select, control, and evaluate external agent frameworks and tools. External frameworks, compilers, terminals, file systems, and execution tools may provide capabilities and perform operations, but they must not replace or supply Culi's core intelligence.

## 2. Established Project Objective

Project Model Culi researches and develops one unified neural model capable of:

- Vietnamese and English language understanding and generation.
- Rust programming comprehension and code generation.
- Dioxus framework comprehension and application.
- General reasoning and problem-solving.
- Planning and autonomous task execution.

## 3. Unified Model Requirement

All core intelligence capabilities must belong to one unified neural model. An external execution mechanism may execute commands or interact with the operating system, but it must not introduce another model responsible for core intelligence.

## 4. Strict Hardware Constraint

The target hardware budget is:

- Ubuntu VPS.
- 4 GB RAM.
- Conventional CPU.
- No dedicated GPU assumed.

The 4 GB RAM target applies to both model training and inference. Any exception requires explicit project owner approval.

Operational definition (D-0005, `docs/TECHNICAL_CONSTRAINTS.md`): maximum 4 GB RAM per training process and per inference process, measured separately, excluding swap, based on peak process memory.

## 5. Architecture Independence

Transformer, RNN, SSM, MoE, recursive networks, and other existing approaches are research subjects, not predetermined solutions. They may be investigated, compared, challenged, combined, or rejected. New architectures may be proposed with technical reasoning and testable hypotheses.

## 6. Current Status

Initialization and experimental foundation are complete through Milestone 9. A standard-library-only development foundation exists under `src/` and `tests/` for repository validation, peak-RSS measurement, experiment running, evaluation, and experimental neural prototypes. Experimental prototypes (feedforward, recurrent, gated-memory, hierarchical-state, compact gated-memory) have been implemented and evaluated on synthetic tasks through Milestone 9. Milestone 9 evaluates cross-task generalization across parity, delayed recall, dependency, multi-recall, and recall-last tasks. No final neural architecture has been selected, and no result constitutes a claim of general intelligence. No packages have been installed, no external data or weights downloaded, no paid APIs called, no VPS configuration modified, Git repository initialized and baseline commit created.

## 7. Unresolved Technical Questions

The following items are TO BE DEFINED:

- Neural architecture family.
- Parameter budget.
- Memory budget breakdown.
- Context-length budget.
- Tokenizer and vocabulary design.
- Training data sources, licenses, and curation policy.
- Synthetic data policy and validation strategy.
- Training methodology.
- Baseline models.
- Evaluation benchmarks and metrics.
- Acceptance thresholds.
- Inference latency and throughput targets.
- Quantization, compression, and deployment strategy.
- Representation of planning and autonomous task execution.
- Safety and alignment requirements.
- Experiment tracking and versioning strategy.
- Data privacy and governance requirements.
- Repository license and contribution policy.
- Team roles and approval authority.

## 8. Document Index

- `README.md`: Project overview and boundaries.
- `.clinerules/project-rules.md`: Project rules.
- `docs/ARCHITECTURE.md`: Architecture status and research subjects.
- `docs/RESEARCH_ROADMAP.md`: Research phases and milestones.
- `docs/TECHNICAL_CONSTRAINTS.md`: Technical and hardware constraints.
- `docs/AGENT_RULES.md`: Agent operating rules.
- `docs/DECISIONS.md`: Decision log.
- `docs/EXPERIMENT_LOG.md`: Experiment index and conventions.
- `research/literature/`: Literature summaries.
- `research/hypotheses/`: Research hypotheses.
- `research/evaluations/`: Evaluation plans.
- `experiments/designs/`: Experiment designs.
- `experiments/results/`: Experiment results.
- `src/`: Minimal standard-library utilities for repository validation, peak-RSS measurement, experiment running, intelligence evaluation, and an experimental neural prototype.
- `tests/`: Validation tests for repository structure, memory measurement, experiment running, evaluation, and the neural prototype.
- `evaluations/`: Structured evaluation datasets.
- `models/`: Saved model parameters and metadata.
- `models/`: Reserved for model artifacts or metadata.
