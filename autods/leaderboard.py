"""Submitting a solution and reading back its score.

The harness shells out to the competition platform's official CLI rather than
talking to a private API, so a reviewer can reproduce a submission by hand.

Set ``submit_to_leaderboard=False`` in the experiment config to run the whole
pipeline without submitting; every other stage is unaffected, and the run record
simply carries no scores.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, List, Optional

from .config import SUBMISSION_FILENAME
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

UNKNOWN_SCORE = "unknown"

#: A float in the CLI output. The original pattern only matched values below 1,
#: which silently dropped the score of every regression task whose metric is a
#: raw error (for example an RMSE of 4.87).
_FLOAT_RE = re.compile(r"\b\d+\.\d+\b")

#: Scores as they appear on a completed submission row.
_COMPLETE_RE = re.compile(r"SubmissionStatus\.COMPLETE,([0-9.]+),([0-9.]+)")


def find_submission_file(competition_name: str, workdir: str = ".") -> Optional[str]:
    """Locate the submission file a solution wrote, searching the likely places."""
    candidates = [
        os.path.abspath(SUBMISSION_FILENAME),
        os.path.join(os.getcwd(), competition_name, SUBMISSION_FILENAME),
        os.path.abspath(os.path.join("..", "input", competition_name, SUBMISSION_FILENAME)),
    ]
    candidates += [
        os.path.join(dirpath, SUBMISSION_FILENAME)
        for dirpath, _dirnames, filenames in os.walk(workdir)
        if SUBMISSION_FILENAME in filenames
    ]

    for path in candidates:
        if os.path.exists(path):
            LOGGER.info("Submission file found at %s", path)
            return path
    return None


def submit(competition_name: str, file_path: str, message: str) -> str:
    """Submit a file to the competition leaderboard and return the CLI output."""
    normalized_path = os.path.abspath(file_path)

    if not os.path.exists(normalized_path):
        return f"Error: submission file {normalized_path} does not exist"

    if os.path.basename(normalized_path).lower() != SUBMISSION_FILENAME:
        return f"Error: only a file named {SUBMISSION_FILENAME} may be submitted"

    command = [
        "kaggle", "competitions", "submit", competition_name,
        "-f", normalized_path, "-m", message,
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode == 0:
            return process.stdout
        return f"Submission failed: {process.stderr}"
    except Exception as exc:
        return f"Submission failed: {exc}"


def latest_submission_row(competition_name: str) -> List[str]:
    """Return the most recent submission row as a list of fields."""
    command = ["kaggle", "competitions", "submissions", "-c", competition_name, "-v"]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return [f"Could not read submissions: {result.stderr}"]

        if result.stdout and "\n" in result.stdout:
            rows = result.stdout.strip().split("\n")
            if len(rows) > 1:  # row 0 is the header
                return rows[1].split()
        return ["No submission history available"]
    except Exception as exc:
        return [f"Could not read submissions: {exc}"]


def parse_metrics(submission_row: List[str]) -> Optional[Dict[str, str]]:
    """Extract the public and private scores from a submission row.

    Returns ``None`` when the submission errored out or no score is present yet.
    """
    if not submission_row:
        return None

    row_text = ",".join(str(field) for field in submission_row)

    if "SubmissionStatus.ERROR" in row_text:
        LOGGER.warning("Submission status is ERROR; no score available")
        return None

    if "SubmissionStatus.COMPLETE" in row_text:
        match = _COMPLETE_RE.search(row_text)
        if match:
            metrics = {"public_score": match.group(1), "private_score": match.group(2)}
            LOGGER.info(
                "Scores: public=%s private=%s",
                metrics["public_score"], metrics["private_score"],
            )
            return metrics

    # No explicit status marker: fall back to the trailing floats on the row.
    scores = _FLOAT_RE.findall(row_text)
    if len(scores) >= 2:
        metrics = {"public_score": scores[-2], "private_score": scores[-1]}
    elif len(scores) == 1:
        metrics = {"public_score": scores[0], "private_score": UNKNOWN_SCORE}
    else:
        return None

    LOGGER.info(
        "Scores (parsed without a status marker): public=%s private=%s",
        metrics["public_score"], metrics["private_score"],
    )
    return metrics
