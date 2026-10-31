# Experiment Log

## Purpose

This document indexes experiments, hypotheses, methods, results, and conclusions for Project Model Culi.

## Status Convention

- `planned`: Designed but not executed.
- `running`: In progress.
- `completed`: Executed and results recorded.
- `validated`: Reviewed and accepted as evidence.
- `rejected`: Reviewed and not accepted as evidence.

## Experiment Runner

A standard-library-only experiment runner is available at `src/experiment_runner.py`. It records experiment ID, timestamp, command, configuration, exit status, runtime, peak RSS, CPU time, CPU utilization, task success, stdout, and stderr. Repeated runs produce an aggregate summary. Throughput and multi-task success rates are planned measurements and are not yet implemented.

Commands:

```text
python3 src/experiment_runner.py -- <command> [args...]
python3 src/experiment_runner.py --repeat N -- <command> [args...]
```

## Evaluation System

An architecture-independent evaluation system is available at `src/evaluation.py` with CLI `src/evaluate.py`. It supports structured datasets, automatic exact/contains scoring, manual review flags, result export, and model-version comparison. A representative dataset is available at `evaluations/model_culi_eval_v1.json`. The mock adapter in `src/mock_model.py` is for testing and demonstration only and is not a Model Culi implementation.

Commands:

```text
python3 src/evaluate.py --dataset evaluations/model_culi_eval_v1.json --mock --output /tmp/scratchpad/mock_eval.json
```

## Experiment Index

| ID | Status | Objective | Method | Key Results |
|---|---|---|---|---|
| sample-dev-validation | completed | Validate the experiment runner and recording pipeline. | Run `python3 -c 'print("sample")'` once. | Runtime: 0.207971 s; peak RSS: 15716352 bytes; CPU time: 0.183460 s; exit status: 0; task success: true. |
| prototype-train-validation | completed | Validate trainable neural prototype and training measurement. | Train the 2-4-1 feedforward prototype on XOR for 5000 epochs. | Initial loss: 0.271442; final loss: 0.000708; parameters changed: true; runtime: 1.393403 s; peak RSS: 15749120 bytes. |
| model-comparison-xor | completed | Compare feedforward and recurrent prototypes on XOR. | Train and evaluate both prototypes across seeds 1-3 for 5000 epochs. | Feedforward: 17 params, mean final loss 0.000755, score 1.000, training runtime mean 1.288737 s, peak RSS max 15876096 bytes. Recurrent: 29 params, mean final loss 0.083496, score 0.833333, training runtime mean 2.271816 s, peak RSS max 15990784 bytes. One recurrent seed failed to learn. |
| sequential-memory-xor-tasks | completed | Test state continuity and generalization on synthetic sequential tasks. | Train feedforward, recurrent, and bounded-recurrent models on parity, delayed recall, and dependency; evaluate at lengths 6 and 8. | Feedforward delayed recall and dependency: 1.000 at length 6, 0.540 at length 8. Recurrent delayed recall/dependency: 1.000 at length 6; dependency 1.000 at length 8. Parity remained near chance for all models. Bounded recurrent was less stable. All runs stayed below 4 GB. |
| memory-organization-synthetic | completed | Test memory organization candidates on delayed recall, dependency, and parity. | Train recurrent, bounded recurrent, gated-memory, and hierarchical models on lengths 6; evaluate lengths 6 and 8 across 3 seeds. | Gated memory: delayed recall 1.000 at lengths 6 and 8; dependency 0.720/0.487. Hierarchical: delayed recall 1.000/0.400; dependency 0.987/0.780. Recurrent: delayed recall 1.000/0.647; dependency 0.787/0.727. Bounded recurrent: less stable. Parity remained near chance. All runs stayed below 4 GB. |
| selective-memory-long-range | completed | Test selective retention and long-range memory. | Train recurrent, gated-memory, and compact gated-memory (hidden size 2) models on parity, delayed recall, dependency, and multi-recall; train lengths 4/6/8; evaluate lengths 4, 6, 8, 10, 12, 16 across 5 seeds. | Recurrent: delayed recall 1.000/1.000/0.948 (len 6/8/16); dependency 0.580/0.612/0.592. Gated memory: delayed recall 1.000/1.000/0.984; dependency 0.724/0.768/0.720. Compact gated memory: delayed recall 1.000/1.000/0.996; dependency 0.484/0.512/0.464. Parity remained near chance for all candidates. All runs stayed below 4 GB. |

## Recording Requirements

Each future experiment entry must include:

- Experiment ID.
- Related hypothesis.
- Objective.
- Method.
- Hardware environment.
- Data used.
- Metrics.
- Results.
- Interpretation.
- Validation status.

## Current Boundary

Experiments through Milestone 8 have been designed and executed on synthetic tasks using experimental prototypes. Results are recorded in this log and in the corresponding summary JSON files. No final architecture has been selected or endorsed, and synthetic-task results are not claims of general intelligence. No packages installed, no external data or weights downloaded.
