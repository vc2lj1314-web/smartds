"""Running a generated program under a wall-clock and CPU budget.

The original script carried two near-duplicate runners, one with a fixed
timeout and CPU limits and one with a dynamic timeout and none. They also
disagreed on how stale a pre-existing ``submission.csv`` had to be before it
counted as freshly written (60 s versus 1 s). They are unified here into one
function whose timeout and tolerance are parameters, and the tolerance is a
single documented constant.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional

from .code_analysis import check_syntax, describe_error, wraps_submission_in_try_except
from .config import (
    SUBMISSION_FILENAME,
    SUBMISSION_MTIME_TOLERANCE_SEC,
    TIMEOUT_SENTINEL,
)
from .logging_utils import get_logger
from .resources import thread_limit_env

LOGGER = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Outcome of one attempt to run a generated program.

    Attributes:
        success: True only if the process exited 0 *and* wrote a fresh
            submission file. An exit code of 0 alone is not enough.
        output: Stdout on success, or the formatted error text on failure.
        execution_time: Wall-clock seconds, or ``None`` if the program never
            started (for example because it failed the pre-flight checks).
        timed_out: True when the harness killed the process.
        submission_path: Path of the freshly written submission, if any.
    """

    success: bool
    output: str
    execution_time: Optional[float] = None
    timed_out: bool = False
    submission_path: Optional[str] = None


def _snapshot_submissions(root: str = ".") -> List[Dict[str, float]]:
    """Record the path and mtime of every existing submission file under ``root``.

    Nothing is deleted: the snapshot is only used afterwards to tell a file this
    run produced from one left behind by a previous run.
    """
    snapshot: List[Dict[str, float]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower() == SUBMISSION_FILENAME:
                path = os.path.join(dirpath, filename)
                snapshot.append({"path": path, "mtime": os.path.getmtime(path)})
    return snapshot


def _find_fresh_submission(
    before: List[Dict[str, float]],
    root: str = ".",
    tolerance: float = SUBMISSION_MTIME_TOLERANCE_SEC,
) -> Optional[str]:
    """Return the path of a submission file written during this run, if any."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower() != SUBMISSION_FILENAME:
                continue
            path = os.path.join(dirpath, filename)
            current_mtime = os.path.getmtime(path)

            is_fresh = True
            for previous in before:
                if previous["path"] == path:
                    is_fresh = (current_mtime - previous["mtime"]) > tolerance
                    break

            if is_fresh:
                return path
    return None


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate a process and its children, escalating to SIGKILL if needed."""
    try:
        process.terminate()
        time.sleep(5)
        if process.poll() is None:
            process.kill()
            time.sleep(2)
    except Exception as exc:
        LOGGER.warning("Error while terminating the process: %s", exc)

    try:
        import psutil  # optional dependency; only needed to reap orphans

        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.terminate()
        time.sleep(2)
        for child in parent.children(recursive=True):
            if child.is_running():
                child.kill()
    except Exception:
        # The parent is already gone, or psutil is unavailable. Either way the
        # main process has been killed, which is what matters.
        pass


def _format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def run_program(
    code: str,
    timeout_seconds: int,
    cpu_limit: int,
    workdir: str = ".",
) -> ExecutionResult:
    """Execute ``code`` in a subprocess under a wall-clock and thread budget.

    Two pre-flight checks run before anything is executed: the program must
    parse, and it must not hide its submission write inside ``try/except``.

    Args:
        code: The program to run.
        timeout_seconds: Wall-clock budget; the process tree is killed after it.
        cpu_limit: Thread budget passed to the numerical libraries.
        workdir: Directory scanned for the submission file.
    """
    if wraps_submission_in_try_except(code):
        return ExecutionResult(
            success=False,
            output=(
                f"Error: the program wraps the {SUBMISSION_FILENAME} write in "
                "try/except, which would hide a failure and leave no submission file."
            ),
        )

    syntax_ok, syntax_error = check_syntax(code)
    if not syntax_ok:
        return ExecutionResult(success=False, output=f"Syntax error: {syntax_error}")

    submissions_before = _snapshot_submissions(workdir)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(code.encode("utf-8"))

    env = thread_limit_env(cpu_limit)
    process: Optional[subprocess.Popen] = None
    timed_out = False
    start_time = time.time()

    LOGGER.info(
        "Executing generated program (timeout %s, %d threads)",
        _format_duration(timeout_seconds),
        cpu_limit,
    )

    try:
        process = subprocess.Popen(
            ["python", temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        def on_timeout() -> None:
            nonlocal timed_out
            if process and process.poll() is None:
                timed_out = True
                LOGGER.warning(
                    "Timeout after %s; terminating the process tree",
                    _format_duration(timeout_seconds),
                )
                _terminate_process_tree(process)

        timer = threading.Timer(timeout_seconds, on_timeout)
        timer.start()

        try:
            stdout, stderr = process.communicate()
        finally:
            timer.cancel()

        execution_time = time.time() - start_time
        os.unlink(temp_path)

        if timed_out:
            return ExecutionResult(
                success=False,
                output=(
                    f"{TIMEOUT_SENTINEL} execution exceeded "
                    f"{_format_duration(timeout_seconds)} and was terminated."
                ),
                execution_time=execution_time,
                timed_out=True,
            )

        if process.returncode != 0:
            return ExecutionResult(
                success=False,
                output=f"Error category: {describe_error(stderr)}\nDetails: {stderr}",
                execution_time=execution_time,
            )

        fresh_submission = _find_fresh_submission(submissions_before, workdir)
        if fresh_submission is None:
            return ExecutionResult(
                success=False,
                output=(
                    "Error: the program finished but no newly written "
                    f"{SUBMISSION_FILENAME} was found. Make sure the program "
                    "writes the submission file."
                ),
                execution_time=execution_time,
            )

        LOGGER.info("Submission written to %s", fresh_submission)
        return ExecutionResult(
            success=True,
            output=stdout,
            execution_time=execution_time,
            submission_path=fresh_submission,
        )

    except Exception:
        if process and process.poll() is None:
            _terminate_process_tree(process)
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        execution_time = time.time() - start_time
        # A crash that coincides with the deadline is reported as a timeout: the
        # process was very likely killed out from under us.
        if timed_out or execution_time >= timeout_seconds * 0.95:
            return ExecutionResult(
                success=False,
                output=(
                    f"{TIMEOUT_SENTINEL} execution exceeded "
                    f"{_format_duration(timeout_seconds)} and was terminated."
                ),
                execution_time=execution_time,
                timed_out=True,
            )
        return ExecutionResult(
            success=False, output=traceback.format_exc(), execution_time=execution_time
        )
