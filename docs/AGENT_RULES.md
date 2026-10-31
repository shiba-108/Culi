# Agent Rules

## 1. Scope

Agents operate only within `/root/culi`.

## 2. Prohibited Actions

Agents must not:

- Implement or train an AI model.
- Install packages.
- Download datasets or model weights.
- Call paid APIs.
- Modify VPS configurations.
- Initialize Git without approval.
- Create, modify, or delete files outside `/root/culi`.
- Treat hypotheses as validated conclusions.

## 3. Required Distinctions

Agents must clearly distinguish:

- Established project objectives.
- Unresolved technical questions.
- Research hypotheses.
- Experimentally validated conclusions.

## 4. Approval Requirements

The following require explicit project owner approval:

- Model implementation.
- Dependency installation.
- Data acquisition.
- Model training.
- Git initialization.
- Hardware constraint exceptions.
- Changes to project scope.

## 5. Documentation Requirements

- All documentation and technical artifacts must be in professional technical English.
- All changes must be recorded in the appropriate document.
- Decisions must be recorded in `docs/DECISIONS.md`.
- Experiments must be recorded in `docs/EXPERIMENT_LOG.md`.

## 6. Agent and Tool Philosophy (D-0006)

Agents must not recreate an existing agent framework unnecessarily. The neural model should possess the intelligence to understand, select, control, and evaluate external agent frameworks and tools. External frameworks, compilers, terminals, file systems, and execution tools may provide capabilities and perform operations, but they must not replace or supply Culi's core intelligence. The model remains responsible for deciding what to do, why to do it, which tools to use, and how to evaluate results.

Performance is a foundational design objective. Research and evaluation must consider intelligence and task success per GB of RAM, CPU inference speed and throughput, end-to-end task completion time, resource consumption per successfully completed task, and accuracy and reliability under constrained resources.

## 7. Reporting Requirements

After initialization or modification, agents must report:

- Complete affected directory tree.
- All created or modified files.
- Summary of each document.
- Confirmation that no files were created outside `/root/culi`.
- Confirmation that no packages were installed or external data downloaded.
- Any unresolved issues.
## 8. Mandatory Documentation Synchronization

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

