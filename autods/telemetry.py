"""Structured record of everything one experiment did.

The original script accumulated this in two module-level dictionaries. Here it
is an object that is created per episode and flushed to disk after every task,
so that a crash halfway through an episode still leaves a usable record.

The resulting JSON is the input to the token-efficiency and attempt-count
analyses reported in the paper.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .logging_utils import get_logger

LOGGER = get_logger(__name__)


class RunRecorder:
    """Collects per-attempt telemetry for one episode and writes it to JSON."""

    def __init__(
        self,
        competitions: List[str],
        generator_model: str,
        node_id: str,
        corrector_model: str,
        episode_tag: str,
    ) -> None:
        """Start a new record.

        Args:
            competitions: Task identifiers this episode covers.
            generator_model: Name of the policy model under evaluation.
            node_id: Opaque label for the machine that produced the record.
            corrector_model: Name of the frozen auxiliary repair model.
            episode_tag: Timestamp string that ties together every artifact of
                this episode (log file, output directory, submission names).
        """
        self.episode_tag = episode_tag
        self.data: Dict[str, Any] = {
            "experiment_info": {
                "start_time": datetime.now().isoformat(),
                "episode_tag": episode_tag,
                "competition_names": competitions,
                "generator_model": generator_model,
                "node_id": node_id,
                "corrector_model": corrector_model,
            },
            "competitions": {},
        }
        self._attempt: Optional[Dict[str, Any]] = None

    # -- attempt lifecycle -------------------------------------------------- #

    def start_attempt(self, attempt_num: int, temperature: float, top_p: float) -> None:
        """Open a new attempt record. Any unsaved previous attempt is dropped."""
        self._attempt = {
            "attempt_number": attempt_num,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "temperature": temperature,
            "top_p": top_p,
            "inference_tokens": 0,
            "correction_tokens": [],
            "result_status": None,
            "submission_info": None,
        }

    def add_inference_tokens(self, tokens: float) -> None:
        """Accumulate tokens spent on the policy model for this attempt."""
        if self._attempt is not None:
            self._attempt["inference_tokens"] += tokens

    def add_correction_tokens(self, tokens: float) -> None:
        """Append the token cost of one auxiliary repair call."""
        if self._attempt is not None:
            self._attempt["correction_tokens"].append(tokens)

    def finish_attempt(
        self,
        status: str,
        submission_message: Optional[str] = None,
        metrics: Optional[Dict[str, str]] = None,
    ) -> None:
        """Close the attempt record with its outcome."""
        if self._attempt is None:
            return
        self._attempt["end_time"] = datetime.now().isoformat()
        self._attempt["result_status"] = status
        if submission_message and metrics:
            self._attempt["submission_info"] = {
                "submit_message": submission_message,
                "metrics": metrics,
            }

    def save_attempt(self, competition_name: str) -> None:
        """Move the current attempt record into the per-task attempt list."""
        if self._attempt is None:
            return
        bucket = self.data["competitions"].setdefault(competition_name, {"attempts": []})
        bucket.setdefault("attempts", []).append(dict(self._attempt))
        self._attempt = None

    # -- task-level ---------------------------------------------------------- #

    def record_error(self, competition_name: str, message: str) -> None:
        """Attach a fatal error message to a task."""
        bucket = self.data["competitions"].setdefault(competition_name, {"attempts": []})
        bucket["error"] = message

    def flush(self, output_path: str) -> None:
        """Write the whole record to ``output_path``, overwriting it."""
        self.data["experiment_info"]["end_time"] = datetime.now().isoformat()
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Run record written to %s", output_path)
