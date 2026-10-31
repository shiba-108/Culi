"""Architecture-independent experiment runner and benchmark harness.

This module records structured experiment results using only the Python standard
library. It measures runtime, peak RSS, and CPU time for a command. Task
success is derived from the process exit status. Throughput and multi-task
success rates are recorded as planned measurements until a workload defines
their units.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentResult:
    experiment_id: str
    timestamp_utc: str
    command: list[str]
    configuration: dict[str, Any]
    status: str
    exit_code: int | None
    runtime_seconds: float
    peak_rss_bytes: int
    cpu_time_seconds: float
    cpu_utilization_percent: float | None
    task_success: bool
    task_success_rate: float | None
    throughput: float | None
    implemented_measurements: list[str]
    planned_measurements: list[str]
    stdout: str
    stderr: str
    error: str | None
    output_dir: str


IMPLEMENTED_MEASUREMENTS = [
    "exit_status",
    "runtime_seconds",
    "peak_rss_bytes",
    "cpu_time_seconds",
    "cpu_utilization_percent",
    "task_success",
]

PLANNED_MEASUREMENTS = [
    "task_success_rate for multi-task suites",
    "throughput (units per second, workload-defined)",
]


def _generate_experiment_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"exp-{timestamp}-{uuid.uuid4().hex[:8]}"


def _read_vmhwm_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError, IndexError, OSError):
        return 0
    return 0


def _run_process(command: list[str], timeout: float | None) -> tuple[int | None, int, float, float, bool, str | None]:
    """Run a command and return status, peak RSS, runtime, CPU time, timeout flag, error."""
    wall_start = time.perf_counter()
    peak_poll = 0
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        runtime = time.perf_counter() - wall_start
        return None, 0, runtime, 0.0, False, str(exc)

    stdout_data = ""
    stderr_data = ""
    timed_out = False
    status = None
    rusage = None

    while True:
        try:
            polled_pid, polled_status, polled_rusage = os.wait4(process.pid, os.WNOHANG)
        except ChildProcessError:
            break
        if polled_pid == process.pid:
            status = polled_status
            rusage = polled_rusage
            break

        peak_poll = max(peak_poll, _read_vmhwm_bytes(process.pid))
        if timeout is not None and time.perf_counter() - wall_start >= timeout:
            timed_out = True
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                _, polled_status, polled_rusage = os.wait4(process.pid, 0)
                status = polled_status
                rusage = polled_rusage
            except ChildProcessError:
                pass
            break
        time.sleep(0.01)

    # Drain output without blocking after process completion.
    try:
        remaining_stdout, remaining_stderr = process.communicate(timeout=1)
        stdout_data += remaining_stdout
        stderr_data += remaining_stderr
    except subprocess.TimeoutExpired:
        process.kill()
        remaining_stdout, remaining_stderr = process.communicate()
        stdout_data += remaining_stdout
        stderr_data += remaining_stderr
        timed_out = True

    runtime = time.perf_counter() - wall_start
    peak_rss = max(peak_poll, _read_vmhwm_bytes(process.pid))
    if rusage is not None:
        peak_rss = max(peak_rss, rusage.ru_maxrss * 1024)
        cpu_time = rusage.ru_utime + rusage.ru_stime
    else:
        cpu_time = 0.0

    if timed_out:
        exit_code = None
        status_name = "timeout"
    elif status is not None and os.WIFEXITED(status):
        exit_code = os.WEXITSTATUS(status)
        status_name = "completed" if exit_code == 0 else "failed"
    elif status is not None and os.WIFSIGNALED(status):
        exit_code = None
        status_name = "signaled"
    else:
        exit_code = process.returncode
        status_name = "completed" if exit_code == 0 else "failed"

    return status_name, exit_code, peak_rss, runtime, cpu_time, timed_out, stdout_data, stderr_data, None


def run_experiment(
    command: list[str],
    experiment_id: str | None = None,
    configuration: dict[str, Any] | None = None,
    output_dir: str | Path = "experiments/results",
    timeout: float | None = None,
) -> ExperimentResult:
    """Run one experiment and store its structured result as JSON."""
    if experiment_id is None:
        experiment_id = _generate_experiment_id()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    configuration = dict(configuration or {})

    status, exit_code, peak_rss, runtime, cpu_time, timed_out, stdout, stderr, error = _run_process(
        command, timeout
    )
    cpu_utilization = (cpu_time / runtime * 100.0) if runtime > 0 else None
    task_success = status == "completed" and exit_code == 0

    result = ExperimentResult(
        experiment_id=experiment_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        command=list(command),
        configuration=configuration,
        status=status,
        exit_code=exit_code,
        runtime_seconds=runtime,
        peak_rss_bytes=peak_rss,
        cpu_time_seconds=cpu_time,
        cpu_utilization_percent=cpu_utilization,
        task_success=task_success,
        task_success_rate=1.0 if task_success else 0.0,
        throughput=None,
        implemented_measurements=list(IMPLEMENTED_MEASUREMENTS),
        planned_measurements=list(PLANNED_MEASUREMENTS),
        stdout=stdout[:4000],
        stderr=stderr[:4000],
        error=error,
        output_dir=str(output_path),
    )

    result_path = output_path / f"{experiment_id}.json"
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    (output_path / f"{experiment_id}.stdout.txt").write_text(stdout, encoding="utf-8")
    (output_path / f"{experiment_id}.stderr.txt").write_text(stderr, encoding="utf-8")
    return result


def run_repeated(
    command: list[str],
    repeat: int,
    experiment_id: str | None = None,
    configuration: dict[str, Any] | None = None,
    output_dir: str | Path = "experiments/results",
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run the same command multiple times and store an aggregate summary."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    base_id = experiment_id or _generate_experiment_id()
    runs = []
    for index in range(1, repeat + 1):
        run_id = f"{base_id}-run-{index}"
        runs.append(
            run_experiment(
                command,
                experiment_id=run_id,
                configuration=configuration,
                output_dir=output_dir,
                timeout=timeout,
            )
        )

    runtimes = [run.runtime_seconds for run in runs]
    peaks = [run.peak_rss_bytes for run in runs]
    successes = sum(1 for run in runs if run.task_success)
    summary = {
        "experiment_id": f"{base_id}-summary",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "configuration": dict(configuration or {}),
        "repeat_count": repeat,
        "run_ids": [run.experiment_id for run in runs],
        "success_count": successes,
        "task_success_rate": successes / repeat,
        "runtime_seconds": {
            "min": min(runtimes),
            "max": max(runtimes),
            "mean": sum(runtimes) / repeat,
        },
        "peak_rss_bytes": {
            "min": min(peaks),
            "max": max(peaks),
            "mean": sum(peaks) / repeat,
        },
        "implemented_measurements": list(IMPLEMENTED_MEASUREMENTS),
        "planned_measurements": list(PLANNED_MEASUREMENTS),
        "output_dir": str(output_dir),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"{base_id}-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def list_results(output_dir: str | Path = "experiments/results") -> list[Path]:
    path = Path(output_dir)
    if not path.exists():
        return []
    return sorted(path.glob("*.json"))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(
            "usage: experiment_runner.py [--id ID] [--repeat N] [--timeout SECONDS] "
            "[--config JSON] [--output-dir DIR] -- COMMAND [ARGS...]",
            file=sys.stderr,
        )
        return 2

    experiment_id = None
    repeat = 1
    timeout = None
    config: dict[str, Any] = {}
    output_dir = "experiments/results"
    command: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--":
            command = argv[index + 1 :]
            break
        if argv[index] == "--id":
            experiment_id = argv[index + 1]
            index += 2
            continue
        if argv[index] == "--repeat":
            repeat = int(argv[index + 1])
            index += 2
            continue
        if argv[index] == "--timeout":
            timeout = float(argv[index + 1])
            index += 2
            continue
        if argv[index] == "--config":
            config = json.loads(argv[index + 1])
            index += 2
            continue
        if argv[index] == "--output-dir":
            output_dir = argv[index + 1]
            index += 2
            continue
        print(f"unknown argument: {argv[index]}", file=sys.stderr)
        return 2
        index += 1

    if not command:
        print("missing command after --", file=sys.stderr)
        return 2

    if repeat == 1:
        result = run_experiment(
            command,
            experiment_id=experiment_id,
            configuration=config,
            output_dir=output_dir,
            timeout=timeout,
        )
        print(f"experiment_id={result.experiment_id}")
        print(f"status={result.status}")
        print(f"exit_code={result.exit_code}")
        print(f"runtime_seconds={result.runtime_seconds:.6f}")
        print(f"peak_rss_bytes={result.peak_rss_bytes}")
        print(f"cpu_time_seconds={result.cpu_time_seconds:.6f}")
        print(f"result_file={output_dir}/{result.experiment_id}.json")
        return 0 if result.task_success else 1

    summary = run_repeated(
        command,
        repeat=repeat,
        experiment_id=experiment_id,
        configuration=config,
        output_dir=output_dir,
        timeout=timeout,
    )
    print(f"experiment_id={summary['experiment_id']}")
    print(f"repeat_count={summary['repeat_count']}")
    print(f"success_count={summary['success_count']}")
    print(f"task_success_rate={summary['task_success_rate']:.3f}")
    print(f"runtime_mean_seconds={summary['runtime_seconds']['mean']:.6f}")
    print(f"peak_rss_max_bytes={summary['peak_rss_bytes']['max']}")
    print(f"summary_file={output_dir}/{summary['experiment_id']}.json")
    return 0 if summary["success_count"] == repeat else 1


if __name__ == "__main__":
    raise SystemExit(main())
