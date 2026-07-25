"""The outer loop: repeated episodes over a set of tasks.

Algorithm 1 describes a *single* trajectory of ``N_meta`` meta-steps over one
task; that is what :mod:`autods.episode` implements, and ``--max-episodes 1`` (the
default) runs exactly that.

``--max-episodes`` above 1 is a harness-level extension outside Algorithm 1: it runs the
whole trajectory again, carrying the best decoding parameters forward through
:class:`ParameterHistory`. It exists for the cross-run parameter study and for
variance estimates. Each repeat restarts the trajectory -- fresh history, fresh
``C*``, fresh controller -- so ``--max-episodes 2 --n-meta 5`` is **not** equivalent
to ``--n-meta 10``: it is two independent 5-step trajectories, not one 10-step
trajectory, and it doubles the number of ``InitialPrompt`` generations.

Telemetry is flushed after every task, so a crash leaves a usable record.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Set

from .config import (
    RECOVERY_TEMPERATURE_BONUS,
    RECOVERY_TOP_P_BONUS,
    ExperimentConfig,
)
from .episode import EpisodeResult, resolve_target, run_episode
from .llm_clients import CorrectorClient, GeneratorClient
from .logging_utils import get_logger
from .param_history import ParameterHistory
from .parameter_learning import ParameterController
from .telemetry import RunRecorder

LOGGER = get_logger(__name__)


def target_reached(
    result: EpisodeResult, target: Optional[float], direction: str
) -> bool:
    """Whether an episode's best score meets the task's reference score."""
    if target is None or result.best_performance is None:
        return False
    score = float(result.best_performance)
    return score >= target if direction == "maximize" else score <= target


def record_recovery_parameters(
    history: ParameterHistory,
    controller_history: List[Dict],
    competition_name: str,
    config: ExperimentConfig,
    episode_start_time: str,
) -> None:
    """Salvage a starting point from an episode that produced no scored submission.

    Two cases are distinguished. If at least one attempt produced a program that
    ran (and merely failed), the lowest temperature among those attempts is
    recorded with a small upward bonus: that region of the space at least
    yields executable code, so the next episode should start near it rather than
    from the generic default.

    If instead every attempt failed at the extraction stage, nothing is
    recorded: no evidence about the sampling parameters was obtained, and
    writing a fabricated best would poison the history.
    """
    if not controller_history:
        LOGGER.info("%s: no attempt history to salvage", competition_name)
        return

    executed = [
        record
        for record in controller_history
        if not record.get("execution_success") and record.get("generation_success")
    ]

    if not executed:
        LOGGER.info(
            "%s: every attempt failed at code extraction; nothing recorded", competition_name
        )
        return

    lowest = min(executed, key=lambda record: record["temperature"])
    temperature = lowest["temperature"] + RECOVERY_TEMPERATURE_BONUS
    top_p = lowest["top_p"] + RECOVERY_TOP_P_BONUS
    LOGGER.info(
        "%s: recording recovery parameters temperature=%.3f top_p=%.3f",
        competition_name, temperature, top_p,
    )

    history.add_episode(
        competition_name=competition_name,
        model=config.generator.model,
        node_id=config.node_id,
        start_time=episode_start_time,
        end_time=datetime.now().isoformat(),
        best_temperature=temperature,
        best_top_p=top_p,
        best_performance=None,
        actual_attempts=len(controller_history),
    )
    history.save()


def run_experiment(config: ExperimentConfig) -> Set[str]:
    """Run the full experiment and return the set of tasks that met their target."""
    if config.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_device)

    generator = GeneratorClient(config.generator)
    corrector = CorrectorClient(config.corrector)
    history = ParameterHistory(config.history_file)

    solved: Set[str] = set()

    LOGGER.info(
        "Starting experiment: %d episode(s) of N_meta meta-steps over %d tasks",
        config.max_episodes, len(config.competitions),
    )

    for episode_index in range(config.max_episodes):
        if len(solved) == len(config.competitions):
            LOGGER.info("Every task has reached its target; stopping early")
            break

        episode_start_time = datetime.now().isoformat()
        episode_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        remaining = [task for task in config.competitions if task not in solved]

        LOGGER.info(
            "=== Episode %d/%d | %d task(s) remaining ===",
            episode_index + 1, config.max_episodes, len(remaining),
        )

        recorder = RunRecorder(
            competitions=config.competitions,
            generator_model=config.generator.model,
            node_id=config.node_id,
            corrector_model=config.corrector.model,
            episode_tag=episode_tag,
        )
        record_path = os.path.join(config.output_root, f"run-record_{episode_tag}.json")

        for competition_name in remaining:
            LOGGER.info("--- Episode %d | task %s ---", episode_index + 1, competition_name)
            target, direction = resolve_target(competition_name)

            try:
                result = run_episode(
                    competition_name=competition_name,
                    config=config,
                    generator=generator,
                    corrector=corrector,
                    history=history,
                    recorder=recorder,
                    episode_tag=episode_tag,
                    episode_start_time=episode_start_time,
                )

                if target_reached(result, target, direction):
                    solved.add(competition_name)
                    LOGGER.info(
                        "%s reached its target (%s %s, achieved %s); it will be skipped from now on",
                        competition_name, target, direction, result.best_performance,
                    )

                has_usable_result = (
                    result.best_temperature is not None
                    and result.best_top_p is not None
                    and (
                        result.best_performance is not None
                        or result.selection_method == "reward"
                    )
                )

                if has_usable_result:
                    history.add_episode(
                        competition_name=competition_name,
                        model=config.generator.model,
                        node_id=config.node_id,
                        start_time=episode_start_time,
                        end_time=datetime.now().isoformat(),
                        best_temperature=result.best_temperature,
                        best_top_p=result.best_top_p,
                        best_performance=result.best_performance,
                        actual_attempts=result.actual_attempts,
                    )
                    history.save()
                else:
                    LOGGER.warning(
                        "%s produced no usable result; falling back to the recovery rule",
                        competition_name,
                    )
                    # The controller lives inside the episode, so its history is
                    # reconstructed from the run record instead.
                    record = recorder.data["competitions"].get(competition_name, {})
                    attempts = [
                        {
                            "temperature": attempt["temperature"],
                            "top_p": attempt["top_p"],
                            "execution_success": attempt["result_status"] == "execution_success",
                            "generation_success": attempt["result_status"] != "extract_failed",
                        }
                        for attempt in record.get("attempts", [])
                    ]
                    record_recovery_parameters(
                        history, attempts, competition_name, config, episode_start_time
                    )

            except Exception as exc:
                LOGGER.error("Task %s failed: %s", competition_name, exc)
                LOGGER.debug(traceback.format_exc())
                recorder.record_error(competition_name, str(exc))

            finally:
                recorder.flush(record_path)

        LOGGER.info("Episode %d complete | solved %d/%d",
                    episode_index + 1, len(solved), len(config.competitions))

    _log_final_summary(config, solved)
    LOGGER.info("Parameter history file: %s", history.file_path)
    return solved


def _log_final_summary(config: ExperimentConfig, solved: Set[str]) -> None:
    """Print the per-task outcome at the end of the experiment."""
    LOGGER.info("Final result: %d/%d tasks reached their target",
                len(solved), len(config.competitions))

    for competition_name in config.competitions:
        target, direction = resolve_target(competition_name)
        status = "reached" if competition_name in solved else "not reached"
        LOGGER.info("  %-55s %s (target %s, %s)", competition_name, status, target, direction)
