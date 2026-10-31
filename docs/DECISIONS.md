# Decisions

## Decision Log

| ID | Status | Decision | Rationale |
|---|---|---|---|
| D-0001 | Approved | Initialize Project Model Culi under `/root/culi`. | Establishes the project workspace and documentation structure. |
| D-0002 | Approved | Adopt the approved directory structure: `README.md`, `.clinerules/project-rules.md`, `docs/`, `research/`, `experiments/`, `src/`, `tests/`, and `models/`. | Provides a minimal initialization scaffold without unnecessary infrastructure. |
| D-0003 | Approved | Record the established unified-model objective and 4 GB RAM CPU-only hardware constraint as project constants. | These are project owner-defined requirements, not implementation decisions. |
| D-0004 | Approved | Treat existing architectures as research subjects rather than predetermined solutions. | Preserves architecture independence. |
| D-0005 | Approved | Adopt the operational definition of the 4GB RAM constraint: maximum 4 GB RAM per training process and per inference process, measured separately, excluding swap, based on peak process memory. | Removes ambiguity in the memory constraint and enables measurable architecture comparison. |
| D-0006 | Approved | Adopt the new strategic direction: discover and develop a new unified neural architecture for strong general intelligence and exceptional resource efficiency on low-end hardware; existing technologies are references, not design boundaries; performance is a foundational design objective; the model natively understands, selects, controls, and evaluates external tools and agent frameworks without unnecessarily recreating an existing agent framework. | Project owner approved a shift from adapting existing architectures to discovering new architecture and efficiency principles. |
| D-0007 | Approved | Implement a minimal standard-library-only development foundation: repository validation and peak-RSS measurement utilities under `src/` and `tests/`. | Supports 4 GB constraint verification and documentation consistency without model implementation, dependencies, or external data. |
| D-0008 | Approved | Implement an architecture-independent intelligence evaluation system with structured datasets, automatic and manual scoring, JSON result export, model-version comparison, and a mock model adapter for testing. | Provides a model-agnostic benchmark harness without neural architecture selection, external APIs, or invented benchmark results. |
| D-0009 | Approved | Implement an experimental feedforward neural prototype, training script, evaluation adapter, and prototype-only XOR dataset for controlled training and inference tests. | Provides a small trainable model for experimentation without claiming a final architecture or language intelligence. |
| D-0010 | Approved | Implement a controlled comparison experiment between the feedforward and recurrent prototypes on the XOR task using deterministic seeds and shared evaluation. | Produces measured evidence on learning stability and resource use without selecting a final architecture. |
| D-0011 | Approved | Implement Milestone 6 sequential memory and reasoning experiment with synthetic tasks, multiple model variants, deterministic seeds, and evaluation at familiar and longer sequence lengths. | Produces measured evidence on state continuity, generalization, and resource use without selecting a final architecture. |
| D-0012 | Approved | Implement Milestone 7 memory organization experiment with gated-memory and hierarchical-state candidates, multiple tasks, deterministic seeds, and familiar/longer evaluation lengths. | Produces measured evidence on memory retention, generalization, stability, and resource use without selecting a final architecture. |
| D-0013 | Approved | Implement Milestone 8 selective memory experiment with gated-memory, compact, and hierarchical candidates, multiple retention distances, interference tests, deterministic seeds, and longer unseen sequences. | Produces measured evidence on selective retention, generalization, stability, and resource use without selecting a final architecture. |
| D-0014 | Approved | Clarify Milestone 8 candidate set: recurrent, gated-memory, and compact gated-memory (hidden size 2). Hierarchical-state was not included in Milestone 8. | Reconciles D-0013 wording with the verified implementation in `src/run_milestone8.py` and `experiments/results/selective_memory_summary.json`. Hierarchical-state was evaluated in Milestone 7. |
| D-0015 | Approved | Initialize Git and adopt GitHub repository `shiba-108/Culi` for version control of Project Model Culi. | Records Git/GitHub adoption; baseline commit created and pushed to `origin/main`. Preserves D-0001 through D-0014. |

## Pending Decisions

The following decisions are TO BE DEFINED:

- Initial research scope.
- Literature review priority.
- Architecture families to investigate first.
- Parameter and memory budgets.
- Tokenization approach.
- Data sources and licenses.
- Baseline models.
- Evaluation metrics and thresholds.
- Experiment tracking tool.
- Version control initialization.
- Project license.
- Safety and alignment policy.

## Change Control

Any new or revised decision must be recorded here with status, rationale, and approval state.
