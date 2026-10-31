"""Peak resident-memory measurement helpers for Model Culi.

This module uses only the Python standard library. It supports the project's
4 GB per-process RAM constraint by measuring the peak resident set size
(VmHWM) of a child process via /proc on Linux.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def _read_vmhwm_bytes(pid: int) -> int:
    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError, IndexError):
        return 0
    return 0


def measure_peak_rss_bytes(command: list[str]) -> int:
    """Run *command* and return its peak resident set size in bytes.

    The measurement polls ``/proc/<pid>/status`` for ``VmHWM`` while the child
    process is running. This reports the peak RSS of that specific child
    process rather than a cumulative value across all children.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    peak_bytes = 0
    while process.poll() is None:
        peak_bytes = max(peak_bytes, _read_vmhwm_bytes(process.pid))
        time.sleep(0.001)
    peak_bytes = max(peak_bytes, _read_vmhwm_bytes(process.pid))
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    return peak_bytes


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: memory_profile.py <command> [args...]", file=sys.stderr)
        return 2
    peak_bytes = measure_peak_rss_bytes(argv)
    print(f"peak_rss_bytes={peak_bytes}")
    print(f"peak_rss_mb={peak_bytes / (1024 * 1024):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
