"""Static checks on generated programs, and classification of error messages.

These functions are pure and side-effect free, which makes them the easiest
part of the harness to unit-test.
"""

from __future__ import annotations

import ast
import re
from typing import Optional, Tuple

from .config import SUBMISSION_FILENAME, TIMEOUT_SENTINEL
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Syntax and structural checks
# --------------------------------------------------------------------------- #


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Return ``(is_valid, error_text)`` for a candidate program."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as exc:
        return False, str(exc)


def wraps_submission_in_try_except(code: str) -> bool:
    """Detect a try/except block that swallows the submission-writing step.

    A generated solution that hides its submission write inside ``try/except``
    can exit with status 0 while producing no file, which would be scored as a
    silent failure. Such programs are rejected before execution.

    The check is textual on purpose: it must also fire on programs that do not
    parse cleanly enough for an AST walk to be informative.
    """
    lines = code.split("\n")
    in_try_block = False
    try_start_line = -1

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("try:"):
            in_try_block = True
            try_start_line = index
            continue

        if in_try_block and (stripped.startswith("except") or stripped.startswith("finally")):
            try_block = "\n".join(lines[try_start_line:index]).lower()
            if any(marker in try_block for marker in (SUBMISSION_FILENAME, "to_csv", "submission")):
                return True
            in_try_block = False

    return False


# --------------------------------------------------------------------------- #
# Deep-learning detection and epoch rewriting
# --------------------------------------------------------------------------- #

DL_IMPORT_MARKERS = (
    "tensorflow", "keras", "torch", "pytorch", "fastai",
    "mxnet", "theano", "caffe", "paddle",
)

DL_API_MARKERS = (
    "model.fit", "fit_generator", "train_loop", "train_on_batch",
    "backward()", "optimizer.step", "DataLoader", "nn.Module",
    "Dense", "Conv2D", "LSTM",
)

EPOCH_PATTERNS = (
    r"epochs\s*=\s*(\d+)",
    r"num_epochs\s*=\s*(\d+)",
    r"n_epochs\s*=\s*(\d+)",
    r"\.fit\([^)]*epochs\s*=\s*(\d+)",
    r"for\s+epoch\s+in\s+range\s*\(\s*(\d+)\s*\)",
)


def detect_dl_code_and_epochs(code: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """Classify a program and, if it trains for many epochs, build a smoke-test variant.

    Returns:
        ``(is_dl_code, original_epochs, smoke_test_code)`` where
        ``smoke_test_code`` is the same program with every epoch count rewritten
        to 1, or ``None`` when no rewrite was needed or possible. Running the
        smoke test first catches pipeline bugs in minutes rather than hours.
    """
    is_dl_code = any(marker in code for marker in DL_IMPORT_MARKERS) or any(
        marker in code for marker in DL_API_MARKERS
    )

    if not is_dl_code:
        return False, None, None

    epochs_value: Optional[int] = None
    smoke_test_code: Optional[str] = None

    for pattern in EPOCH_PATTERNS:
        matches = re.findall(pattern, code)
        if not matches:
            continue

        epochs_value = int(matches[0])
        if epochs_value > 1:
            smoke_test_code = code
            for match in matches:
                if int(match) <= 1:
                    continue
                replacements = (
                    (f"epochs={match}", "epochs=1"),
                    (f"epochs = {match}", "epochs = 1"),
                    (f"num_epochs={match}", "num_epochs=1"),
                    (f"num_epochs = {match}", "num_epochs = 1"),
                    (f"n_epochs={match}", "n_epochs=1"),
                    (f"n_epochs = {match}", "n_epochs = 1"),
                    (f"range({match})", "range(1)"),
                    (f"range({match},", "range(1,"),
                )
                for old, new in replacements:
                    smoke_test_code = smoke_test_code.replace(old, new)
        break

    return True, epochs_value, smoke_test_code


def restore_epoch_count(code: str, original_epochs: int) -> str:
    """Undo :func:`detect_dl_code_and_epochs`'s rewrite, restoring the epoch count."""
    patterns = (
        (r"epochs\s*=\s*1", f"epochs={original_epochs}"),
        (r"epochs\s*=\s*1,", f"epochs={original_epochs},"),
        (r"num_epochs\s*=\s*1", f"num_epochs={original_epochs}"),
        (r"num_epochs\s*=\s*1,", f"num_epochs={original_epochs},"),
        (r"n_epochs\s*=\s*1", f"n_epochs={original_epochs}"),
        (r"n_epochs\s*=\s*1,", f"n_epochs={original_epochs},"),
        (r"range\s*\(\s*1\s*\)", f"range({original_epochs})"),
        (r"range\s*\(\s*1\s*,", f"range({original_epochs},"),
    )
    restored = code
    for pattern, replacement in patterns:
        restored = re.sub(pattern, replacement, restored)
    return restored


# --------------------------------------------------------------------------- #
# Error message classification
# --------------------------------------------------------------------------- #

ERROR_DESCRIPTIONS = {
    "ImportError": "import error - a required library or module is missing",
    "ModuleNotFoundError": "import error - a required library or module is missing",
    "TypeError": "type error - wrong argument type or incompatible operation",
    "ValueError": "value error - correct type but inappropriate value",
    "AttributeError": "attribute error - the object has no such attribute or method",
    "IndexError": "index error - list index out of range",
    "KeyError": "key error - the key is not present in the mapping",
    "FileNotFoundError": "file not found - wrong path or missing data file",
    "PermissionError": "permission error - insufficient rights to access a file or resource",
    "NameError": "name error - an undefined variable was used",
    "SyntaxError": "syntax error - the program does not parse",
    "MemoryError": "memory error - out of memory",
    "RuntimeError": "runtime error - generic runtime problem",
    "ZeroDivisionError": "division by zero",
}


def describe_error(error_message: str) -> str:
    """Map a traceback to a short, human-readable category label."""
    for error_type, description in ERROR_DESCRIPTIONS.items():
        if error_type in error_message:
            return description
    return "general error - needs further analysis"


#: Free-text markers that indicate a timeout in messages the harness did not
#: produce itself, e.g. a supervisor that killed the process. The primary
#: detector is the ``TIMEOUT_SENTINEL`` prefix emitted by :mod:`autods.execution`.
TIMEOUT_TEXT_MARKERS = (
    "timeout",
    "timed out",
    "time limit exceeded",
    "execution time limit",
    "process killed",
    "killed -9",
)


def is_timeout_error(error_message: Optional[str]) -> bool:
    """Return True when the message describes a timeout rather than a code defect.

    Detection is anchored on the machine-readable sentinel the runner emits.
    The free-text markers are a fallback for messages produced elsewhere; they
    are kept deliberately specific, because a loose marker (for instance the
    bare string ``epochs=1``) would misclassify ordinary tracebacks as timeouts
    and abort the repair loop early.
    """
    if not error_message:
        return False
    if TIMEOUT_SENTINEL in error_message:
        return True
    lowered = error_message.lower()
    return any(marker in lowered for marker in TIMEOUT_TEXT_MARKERS)
