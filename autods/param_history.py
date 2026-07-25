"""Persistent memory of the best sampling parameters found per (task, model).

This is what makes the search cumulative across episodes: an episode starts from
the best parameters any previous episode found for the same task and model,
nudged slightly downwards so that upward exploration remains possible.

The store is a single JSON file with two sections: an append-only ``episodes``
log for analysis, and a ``best_parameters`` index used at start-up.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from .config import BEST_HISTORICAL_PARAMS_FILE, COMPETITION_TARGETS, DEFAULT_DIRECTION
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def is_better(new_score: Any, old_score: Any, competition_name: str) -> bool:
    """Compare two scores under the task's own optimisation direction."""
    if new_score is None:
        return False
    if old_score is None:
        return True

    _target, direction = COMPETITION_TARGETS.get(competition_name, (None, DEFAULT_DIRECTION))

    try:
        new_value = float(new_score)
        old_value = float(old_score)
    except (TypeError, ValueError):
        return False

    return new_value > old_value if direction == "maximize" else new_value < old_value


class ParameterHistory:
    """JSON-backed record of past episodes and of the best parameters per task."""

    def __init__(self, file_path: str = BEST_HISTORICAL_PARAMS_FILE) -> None:
        self.file_path = file_path
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        empty: Dict[str, Any] = {"episodes": [], "best_parameters": []}
        if not os.path.exists(self.file_path):
            return empty
        try:
            with open(self.file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            LOGGER.warning("Could not read %s (%s); starting fresh", self.file_path, exc)
            return empty

        data.setdefault("episodes", [])
        data.setdefault("best_parameters", [])
        return data

    def save(self) -> None:
        """Write the store back to disk."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
            LOGGER.info("Parameter history saved to %s", self.file_path)
        except Exception as exc:
            LOGGER.error("Could not save parameter history: %s", exc)

    def best_parameters(
        self, competition_name: str, model: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return the best ``(temperature, top_p)`` on record, or ``(None, None)``."""
        for entry in self.data["best_parameters"]:
            if entry["competition_name"] == competition_name and entry["model"] == model:
                return entry["best_temperature"], entry["best_top_p"]
        return None, None

    def add_episode(
        self,
        competition_name: str,
        model: str,
        node_id: str,
        start_time: str,
        end_time: str,
        best_temperature: float,
        best_top_p: float,
        best_performance: Optional[float],
        actual_attempts: int,
    ) -> None:
        """Append one episode to the log and refresh the best-parameters index."""
        self.data["episodes"].append(
            {
                "competition_name": competition_name,
                "model": model,
                "node_id": node_id,
                "start_time": start_time,
                "end_time": end_time,
                "best_temperature": best_temperature,
                "best_top_p": best_top_p,
                "best_performance": best_performance,
                "actual_attempts": actual_attempts,
            }
        )
        self._update_best(
            competition_name, model, best_temperature, best_top_p, best_performance
        )

    def _update_best(
        self,
        competition_name: str,
        model: str,
        best_temperature: float,
        best_top_p: float,
        best_performance: Optional[float],
    ) -> None:
        """Replace the stored best for this (task, model) if the new one wins."""
        index = -1
        for position, entry in enumerate(self.data["best_parameters"]):
            if entry["competition_name"] == competition_name and entry["model"] == model:
                index = position
                break

        if index == -1:
            should_update = True
        else:
            should_update = is_better(
                best_performance,
                self.data["best_parameters"][index]["best_performance"],
                competition_name,
            )

        if not should_update:
            return

        record = {
            "competition_name": competition_name,
            "model": model,
            "best_temperature": best_temperature,
            "best_top_p": best_top_p,
            "best_performance": best_performance,
        }
        if index == -1:
            self.data["best_parameters"].append(record)
            LOGGER.info("Recorded first best parameters for %s / %s", competition_name, model)
        else:
            self.data["best_parameters"][index] = record
            LOGGER.info("Updated best parameters for %s / %s", competition_name, model)
