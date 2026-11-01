# Project Model Culi Rules

## 0. New Session Context Restoration

At the beginning of every new session, 108 must:

1. Read `.clinerules/project-rules.md` first.
2. Read `PROJECT_HANDOVER.md`.
3. Read `docs/PROJECT_MASTER.md`.
4. Read `docs/DECISIONS.md`.
5. Read `docs/ARCHITECTURE.md`.
6. Read `docs/TECHNICAL_CONSTRAINTS.md`.
7. Read `docs/EXPERIMENT_LOG.md`.
8. Inspect actual Git branch, commit, working-tree status, and unfinished changes.
9. Reconcile the documents with actual source code and experiment artifacts.
10. Report a concise context-restoration summary before accepting new implementation work.

Do not duplicate detailed experiment results in this file. Reference `PROJECT_HANDOVER.md` and the relevant technical documents.

## 1. Workspace Rule

All project-owned documentation, research records, source code, and experiment records must remain under `/root/culi`.

## 2. Documentation Language Rule

All documentation and technical artifacts must be written in professional technical English. Vietnamese may be used only in communication with the project owner.

## 3. Established Objective

Project Model Culi aims to research and develop one unified neural model capable of:

- Vietnamese and English language understanding and generation.
- Rust programming comprehension and code generation.
- Dioxus framework comprehension and application.
- General reasoning and problem-solving.
- Planning and autonomous task execution.

## 4. Unified Model Rule

All core intelligence capabilities must belong to one unified neural model. An external execution mechanism may execute commands or interact with the operating system, but it must not introduce another model responsible for core intelligence.

## 5. Hardware Constraint

The target hardware budget is:

- Ubuntu VPS.
- 4 GB RAM.
- Conventional CPU.
- No dedicated GPU assumed.

The 4 GB RAM target applies to both model training and inference. Any exception requires explicit project owner approval.

## 6. Architecture Independence Rule

Existing approaches, including Transformer, RNN, SSM, MoE, and recursive networks, are research subjects rather than predetermined solutions. They may be investigated, compared, challenged, combined, or rejected. New architectures require technical reasoning and testable hypotheses.

## 7. Initialization Restrictions

During initialization and until explicitly authorized:

- Do not implement or train any model.
- Do not install packages.
- Do not download datasets or model weights.
- Do not call paid APIs.
- Do not modify VPS configurations.
- Do not initialize Git without further approval.

## 8. Approval Gates

Implementation, dependency installation, data acquisition, model training, and Git initialization require explicit project owner approval.

## 9. Classification Discipline

Documentation must clearly distinguish:

- Established project objectives.
- Unresolved technical questions.
- Research hypotheses.
- Experimentally validated conclusions.
## 10. Mandatory Documentation Synchronization

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

## 11. Additional Constraints and Approved Decisions

- Approved decisions D-0001 through D-0016 are recorded in `docs/DECISIONS.md` and must be preserved.
- No Docker or container orchestration may be introduced without explicit project-owner approval.
- No external LLM or API may serve as the core intelligence; all core intelligence must reside in the unified neural model.
- No neural architecture is predetermined; Transformer, RNN, SSM, MoE, recursive networks, and other approaches remain research subjects.
- Git safety: never force-push, overwrite remote history, or discard project data.
- Owner approval is required before starting any new milestone.
- Do not claim that unimplemented features or capabilities are complete.
- The 4 GB per-process RAM limit and CPU-only target remain in force.

