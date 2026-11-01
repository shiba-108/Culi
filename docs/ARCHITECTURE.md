# Architecture

## 1. Status

No architecture has been selected. No implementation exists.

## 1.1 Strategic Direction (D-0006)

Project Model Culi now aims to discover and develop a new unified neural architecture for strong general intelligence and exceptional resource efficiency on low-end hardware. Existing technologies are references, not design boundaries. Performance is a foundational design objective, not a later optimization.

The model must natively understand, select, control, and evaluate external tools and agent frameworks. External frameworks, compilers, terminals, file systems, and execution tools may provide capabilities and perform operations, but they must not replace or supply Culi's core intelligence.

## 1.2 Experimental Prototype Status

A small experimental feedforward neural prototype has been implemented for controlled training, saving, loading, and inference tests. It is an experimental candidate only, not a final architecture decision and not a claim of intelligence capability. A controlled comparison with an experimental recurrent prototype has also been executed on the XOR task. A sequential memory and reasoning experiment (Milestone 6) has been executed on synthetic tasks with feedforward, recurrent, and bounded-recurrent candidates. A memory organization experiment (Milestone 7) has been executed with recurrent, bounded-recurrent, gated-memory, and hierarchical-state candidates. A selective memory experiment (Milestone 8) has been executed with recurrent, gated-memory, and compact gated-memory candidates on parity, delayed recall, dependency, and multi-recall tasks at lengths up to 16. A cross-task generalization experiment (Milestone 9) has been executed with recurrent, gated-memory, compact gated-memory, and hierarchical-state candidates across five synthetic tasks and multiple train/test lengths. No architecture has been selected or endorsed.

## 2. Principles

- Core intelligence must reside in one unified neural model.
- The model must target a 4 GB RAM, CPU-only Ubuntu VPS.
- Existing architectures are research subjects, not predetermined solutions.
- Architectural proposals require technical reasoning and testable hypotheses.

## 3. Research Subjects

The following are subjects for investigation, comparison, challenge, combination, or rejection:

- Transformer variants.
- Recurrent neural networks.
- State-space models.
- Mixture-of-experts designs.
- Recursive networks.
- Hybrid architectures.
- Novel architectures not represented by existing families.

## 4. Constraints

- Training and inference must respect the 4 GB RAM target unless an exception is explicitly approved.
- No dedicated GPU may be assumed.
- The architecture must support the established capabilities: Vietnamese/English language, Rust/Dioxus code, reasoning, and planning/task selection.
- An external execution mechanism may perform operations but must not provide core intelligence.

## 5. Unresolved Questions

The following are TO BE DEFINED:

- Selected architecture family.
- Layer and block design.
- Parameter budget.
- Memory allocation strategy.
- Context-length mechanism.
- Tokenization approach.
- Training objective and loss design.
- Inference procedure.
- Quantization and compression approach.
- Planning and tool-use representation.
- Evaluation-driven architecture refinement process.

## 6. Non-Decision Statement

This document does not select, endorse, or implement any architecture.
