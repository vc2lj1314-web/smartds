"""CPU budget control.

Every generated program is given the same, deliberately small CPU budget so
that runtime comparisons between configurations are meaningful and so that one
runaway solution cannot starve the rest of the machine. The budget is enforced
in two ways: thread-count environment variables for the numerical libraries,
and (on Linux) a CPU affinity mask.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Dict, Optional

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

#: Environment variables honoured by the numerical stacks a generated solution
#: is likely to import. Every one is set to the same thread budget.
THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "SKLEARN_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "LIGHTGBM_NUM_THREADS",
    "CATBOOST_NUM_THREADS",
)


def thread_limit_env(cpu_limit: int, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a copy of ``base_env`` with every thread-count variable capped.

    Args:
        cpu_limit: Maximum number of threads any single library may spawn.
        base_env: Environment to start from; defaults to the current one.
    """
    env = dict(base_env if base_env is not None else os.environ)
    for key in THREAD_LIMIT_VARS:
        env[key] = str(cpu_limit)
    return env


def apply_process_cpu_limits(cpu_limit: int, pin_affinity: bool = True) -> None:
    """Apply the CPU budget to the current process.

    Sets the thread-count variables in ``os.environ`` (so that they are also
    inherited by child processes) and, on Linux, pins the process to the first
    ``cpu_limit`` cores.
    """
    for key, value in thread_limit_env(cpu_limit).items():
        if key in THREAD_LIMIT_VARS:
            os.environ[key] = value
    LOGGER.info("Thread limits set to %d for %d libraries", cpu_limit, len(THREAD_LIMIT_VARS))

    if not pin_affinity:
        return

    try:
        if hasattr(os, "sched_setaffinity"):
            cores = set(range(cpu_limit))
            os.sched_setaffinity(0, cores)
            LOGGER.info("CPU affinity pinned to cores %s", sorted(cores))
        else:
            LOGGER.info("CPU affinity not supported on this platform; skipping")
    except Exception as exc:  # non-fatal: thread limits are the primary control
        LOGGER.warning("Could not set CPU affinity (ignored): %s", exc)


def describe_cpu_limits(cpu_limit: int) -> str:
    """Return a short human-readable report of the effective CPU limits."""
    lines = [f"Detected CPU cores: {multiprocessing.cpu_count()}"]
    for key in THREAD_LIMIT_VARS:
        value = os.environ.get(key, "<unset>")
        marker = "ok" if value == str(cpu_limit) else "!!"
        lines.append(f"  [{marker}] {key}={value}")

    try:
        if hasattr(os, "sched_getaffinity"):
            lines.append(f"CPU affinity: {sorted(os.sched_getaffinity(0))}")
        else:
            lines.append("CPU affinity: unsupported on this platform")
    except Exception as exc:
        lines.append(f"CPU affinity: unavailable ({exc})")

    return "\n".join(lines)
