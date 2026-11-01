# Project Model Culi

## Status

Initialization and experimental foundation complete through Milestone 10. Small experimental neural prototypes (feedforward, recurrent, gated-memory, hierarchical-state, compact gated-memory) and synthetic-task experiments have been implemented and executed under `src/` and `experiments/`. Milestone 10 evaluates collaborative specialist intelligence on composite tasks. No final neural architecture has been selected, and synthetic-task results are experimental evidence only, not claims of general intelligence. No packages have been installed, no datasets or model weights have been downloaded, no paid APIs have been called, no VPS configuration has been modified, and Git repository initialized; baseline commit created.

## Established Project Objective

Project Model Culi researches and develops one unified neural model capable of:

- Vietnamese and English language understanding and generation.
- Rust programming comprehension and code generation.
- Dioxus framework comprehension and application.
- General reasoning and problem-solving.
- Planning and autonomous task execution.

An external execution mechanism may execute commands or interact with the operating system, but it must not introduce another model responsible for core intelligence.

## Strategic Direction (D-0006)

Project Model Culi now aims to discover and develop a new unified neural architecture for strong general intelligence and exceptional resource efficiency on low-end hardware. Existing technologies are references, not design boundaries. Performance is a foundational design objective, not a later optimization.

### Six Foundational Principles

1. New Architecture: Discover and develop potentially new neural architectures rather than assuming existing architecture families define the available solutions.
2. Unified Intelligence: One neural model owns all core intelligence, including knowledge application, understanding, reasoning, planning, decision-making, and tool selection.
3. High Intelligence: Support Vietnamese, English, Rust, Dioxus, general reasoning, problem-solving, and autonomous task execution.
4. Maximum Efficiency: Maximize useful intelligence and task capability per unit of computational resources.
5. High Performance: Prioritize compact model size, fast response, high throughput, low latency, and efficient RAM and CPU utilization.
6. Break Limits: Challenge existing technical limitations and explore fundamentally new solutions. Scientific and experimental evidence must validate every claim.

### Agent and Tool Philosophy

Culi must not recreate an existing agent framework unnecessarily. The neural model should possess the intelligence to understand, select, control, and evaluate external agent frameworks and tools. External frameworks, compilers, terminals, file systems, and execution tools may provide capabilities and perform operations, but they must not replace or supply Culi's core intelligence. The model remains responsible for deciding what to do, why to do it, which tools to use, and how to evaluate results.

## Strict Hardware Constraint

The target hardware budget is:

- Ubuntu VPS.
- 4 GB RAM.
- Conventional CPU.
- No dedicated GPU assumed.

The 4 GB RAM target applies to both model training and inference. Any exception requires explicit project owner approval.

Operational definition: maximum 4 GB RAM per training process and per inference process, measured separately, excluding swap, based on peak process memory (see `docs/TECHNICAL_CONSTRAINTS.md`, D-0005).

## Architecture Independence

Transformer, RNN, SSM, MoE, recursive networks, and other existing approaches are research subjects, not predetermined solutions. They may be investigated, compared, challenged, combined, or rejected. Entirely new architectures may be proposed when supported by technical reasoning and testable hypotheses.

## Repository Layout

```text
/root/culi/
├── README.md
├── .clinerules/
│   └── project-rules.md
├── docs/
├── research/
├── experiments/
├── src/
├── tests/
└── models/
```

## Development Foundation

A minimal standard-library-only development foundation is present:

- `src/memory_profile.py`: measures peak RSS of a child process for 4 GB constraint checks.
- `src/repo_check.py`: validates repository structure and documentation consistency.
- `src/experiment_runner.py`: runs commands, records structured experiment results, and supports repeated runs.
- `src/evaluation.py`: architecture-independent evaluation engine with datasets, scoring, result export, and comparison.
- `src/mock_model.py`: mock model adapter for testing and demonstration only.
- `src/evaluate.py`: command-line interface for evaluation runs.
- `src/compare_models.py`: controlled comparison of feedforward and recurrent prototypes.
- `src/sequential_tasks.py`: synthetic sequential task generation.
- `src/train_sequential.py`: trains models on sequential tasks.
- `src/sequential_adapters.py`: evaluation adapters for sequential models.
- `src/run_milestone6.py`: runs the sequential memory experiment.
- `src/memory_model.py`: experimental gated-memory neural prototype.
- `src/hierarchical_model.py`: experimental hierarchical-state neural prototype.
- `src/run_milestone7.py`: runs the memory organization experiment.
- `src/run_milestone8.py`: runs the selective memory experiment.
- `src/run_milestone9.py`: runs the cross-task generalization experiment.
- `src/run_milestone10.py`: runs the collaborative specialist intelligence experiment.
- `src/neural_model.py`: small experimental feedforward neural network prototype.
- `src/train_prototype.py`: trains the neural prototype on a controlled XOR task.
- `src/prototype_adapter.py`: evaluation adapter for the neural prototype.
- `evaluations/model_culi_eval_v1.json`: representative evaluation dataset.
- `evaluations/model_culi_prototype_xor_v1.json`: prototype-only XOR evaluation dataset.
- `tests/test_repo_structure.py`: repository and documentation consistency tests.
- `tests/test_memory_profile.py`: memory measurement tests.
- `tests/test_experiment_runner.py`: experiment runner tests.
- `tests/test_evaluation.py`: evaluation system tests.
- `tests/test_neural_model.py`: neural prototype tests.
- `tests/test_prototype_adapter.py`: prototype adapter tests.

Commands:

```text
python3 -m unittest discover -s tests
python3 src/repo_check.py
python3 src/train_prototype.py --epochs 5000
python3 src/evaluate.py --dataset evaluations/model_culi_prototype_xor_v1.json --adapter prototype_adapter:PrototypeModelAdapter --model-path models/prototype_xor.json --output /tmp/scratchpad/prototype_eval.json
python3 src/compare_models.py
python3 src/run_milestone7.py
python3 src/run_milestone8.py
python3 src/run_milestone9.py
python3 src/run_milestone10.py
```

## Documentation Policy

All project documentation and technical artifacts must be written in professional technical English.

## Current Boundaries

- No final neural architecture has been selected; experimental prototypes only.
- Experiments use synthetic tasks; no claim of general intelligence or language capability.
- No packages have been installed.
- No datasets or model weights have been downloaded.
- No paid APIs have been called.
- No VPS configuration has been modified.
- Git repository initialized; baseline commit created.

## Next Step

Documentation reconciliation complete. Further research planning, literature review, and any Milestone 9 scope require project owner approval. Model implementation beyond approved experimental prototypes remains prohibited until explicitly authorized.
