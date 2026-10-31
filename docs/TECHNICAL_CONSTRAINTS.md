# Technical Constraints

## Hardware Constraints

- Operating system target: Ubuntu VPS.
- RAM target: 4 GB.
- Processor: Conventional CPU.
- GPU: None assumed.
- The 4 GB RAM target applies to both model training and inference.

## Resource Policy

Any proposed exception to the hardware constraint requires explicit project owner approval. Temporary or convenience-based assumptions of higher-memory or GPU-equipped machines are not permitted.

## Performance Philosophy (D-0006)

Performance is a foundational design objective, not a later optimization task. Future research and evaluation should consider:

- Intelligence and task success per GB of RAM.
- CPU inference speed and throughput.
- End-to-end task completion time.
- Resource consumption per successfully completed task.
- Accuracy and reliability under constrained resources.

Comparisons must use fair and reproducible evaluation conditions.

## Software and Runtime Constraints

The following are TO BE DEFINED:

- Programming languages for research tooling.
- Dependency management policy.
- Supported runtime versions.
- Reproducibility environment.
- Storage limits.
- Artifact retention policy.

## Data Constraints

- No datasets have been downloaded.
- No model weights have been downloaded.
- Data acquisition requires explicit approval.
- Data licenses and privacy requirements are TO BE DEFINED.

## Implementation Constraints

- No model has been implemented.
- No training has been performed.
- No packages have been installed.
- No VPS configuration has been modified.

## Operational Definition of the 4GB RAM Constraint

The following operational definition is established as an official project rule (see `docs/DECISIONS.md`, D-0005):

- Maximum 4 GB RAM per training process.
- Maximum 4 GB RAM per inference process.
- Training and inference peak memory are measured separately.
- Swap usage is excluded from the 4 GB limit.
- Peak process memory is measured, not average usage.
- The limit applies to the peak resident memory of the relevant process, including model parameters, gradients, optimizer states, activations, runtime buffers, and framework overhead attributable to that process.

The following remain TO BE DEFINED and require project-owner confirmation:

- Whether the 4 GB refers to total physical RAM or memory available to the process after OS reservation.
- Whether training and inference processes may coexist on the same machine.
- The acceptable safety headroom percentage (if any).
- Treatment of multi-process data loaders and framework allocator caches.

Any exception must be recorded in `docs/DECISIONS.md` with rationale, scope, and approval status.

## Exception Process


Any exception must be recorded in `docs/DECISIONS.md` with rationale, scope, and approval status.
