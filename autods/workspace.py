"""Where an episode writes its artifacts.

Every episode gets its own directory named ``<task>_<episode_tag>``, so that
repeated episodes over the same task never overwrite each other and every file
can be traced back to the episode that produced it. The layout is:

``<task>_<tag>/raw_response_attempt_<n>.txt``
    Unmodified model response.
``<task>_<tag>/extracted_code_attempt_<n>.py``
    Program as extracted, before any repair.
``<task>_<tag>/corrected_code_attempt_<n>.py``
    Program after the self-correction loop, i.e. what actually ran.
``<task>_<tag>/parameter_log.json``
    Per-attempt parameters, rewards and the selected best configuration.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .logging_utils import get_logger

LOGGER = get_logger(__name__)


class EpisodeWorkspace:
    """Creates and writes into one episode's artifact directory."""

    def __init__(self, competition_name: str, episode_tag: str, root: str = ".") -> None:
        self.competition_name = competition_name
        self.episode_tag = episode_tag
        self.path = os.path.join(root, f"{competition_name}_{episode_tag}")
        os.makedirs(self.path, exist_ok=True)

    def _write(self, filename: str, content: str) -> str:
        full_path = os.path.join(self.path, filename)
        with open(full_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return full_path

    def save_response(self, stage: str, attempt_num: int, content: str) -> str:
        """Store a raw model response. ``stage`` is e.g. ``initial`` or ``optimized``."""
        path = self._write(f"{stage}_response_attempt_{attempt_num}.txt", content)
        LOGGER.info("Saved response to %s", path)
        return path

    def save_code(self, stage: str, attempt_num: int, code: str) -> str:
        """Store a program at some stage of the pipeline."""
        path = self._write(f"{stage}_code_attempt_{attempt_num}.py", code)
        LOGGER.info("Saved program to %s", path)
        return path

    def save_parameter_log(self, payload: Dict[str, Any]) -> str:
        """Store the controller's per-attempt history for this episode."""
        full_path = os.path.join(self.path, "parameter_log.json")
        with open(full_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Saved parameter log to %s", full_path)
        return full_path

    def archive_submission(self, submission_path: str, attempt_num: int, timestamp: str) -> Optional[str]:
        """Rename a submitted file so that it is not mistaken for a fresh one.

        The runner decides whether a program produced a submission by comparing
        modification times, so a submitted file has to be moved out of the way
        before the next attempt runs.
        """
        target_name = (
            f"submission_{self.competition_name}_{self.episode_tag}"
            f"_attempt_{attempt_num}_{timestamp}.csv"
        )
        target_path = os.path.join(os.getcwd(), target_name)
        try:
            os.replace(submission_path, target_path)
            LOGGER.info("Archived submission as %s", target_path)
            return target_path
        except Exception as exc:
            LOGGER.error("Could not archive the submission file: %s", exc)
            return None
