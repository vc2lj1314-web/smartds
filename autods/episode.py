"""One task, run as the meta-step loop of Algorithm 1.

``Episode.run`` is the ``for t = 1 to N_meta`` loop. Each meta-step:

1. builds the state ``S_t`` from the history (Eq. 3) and asks SATE for a
   trajectory action ``a_t``, a prompt ``Q_t`` and a decoding configuration
   ``theta_t`` -- except at ``t = 1``, which uses ``InitialPrompt(D)`` and
   ``theta_0``;
2. produces the meta-step's first candidate ``C_t,0`` -- from ``LocalRepair``
   when ``a_t = repair``, otherwise from the domain-aligned reasoning model
   ``M_DS(Q_t; theta_t)``;
3. hands the candidate to the execution-guided refinement loop, which runs the
   inner ``for k = 0 to N_repair`` loop;
4. records the outcome in ``H``, updates ``C*`` and ``P_best`` if the candidate
   executed and scored better, and returns immediately on ``MeetTarget``.

The best-so-far solution and the history span all ``N_meta`` meta-steps: they
are properties of the trajectory, not of any one meta-step.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .code_extraction import extract_fenced_code, extract_program
from .config import (
    COMPETITION_TARGETS,
    DEFAULT_DIRECTION,
    HISTORICAL_TEMPERATURE_BACKOFF,
    HISTORICAL_TOP_P_BACKOFF,
    SUBMISSION_FILENAME,
    SUBMISSION_POLL_DELAY_SEC,
    TEMPERATURE_HARD_MIN,
    TOP_P_HARD_MIN,
    ExperimentConfig,
)
from .correction_graph import self_correct_and_execute
from .leaderboard import (
    UNKNOWN_SCORE,
    find_submission_file,
    latest_submission_row,
    parse_metrics,
    submit,
)
from .llm_clients import CorrectorClient, GeneratorClient
from .logging_utils import get_logger
from .param_history import ParameterHistory, is_better
from .parameter_learning import ParameterController
from .prompt_builders import (
    build_error_informed_task_prompt,
    build_generic_correction_prompt,
    build_optimization_prompt,
    build_regeneration_prompt,
    summarise_errors,
)
from .prompts import code_correction_system_prompt, generation_system_prompt, optimization_system_prompt
from .sate import build_state, select_action
from .telemetry import RunRecorder
from .workspace import EpisodeWorkspace

LOGGER = get_logger(__name__)


@dataclass
class EpisodeResult:
    """What one episode contributes to the cross-episode parameter history."""

    best_temperature: Optional[float]
    best_top_p: Optional[float]
    best_performance: Optional[float]
    actual_attempts: int
    selection_method: str  # "performance" or "reward"


def load_task_prompt(prompts_path: str, competition_name: str) -> str:
    """Read one task's natural-language description from the prompt file."""
    with open(prompts_path, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    for item in prompts:
        if item["competition_name"] == competition_name:
            return item["prompt"]

    raise ValueError(f"Task '{competition_name}' not found in {prompts_path}")


def resolve_target(competition_name: str) -> Tuple[Optional[float], str]:
    """Look up the task's reference score and optimisation direction."""
    return COMPETITION_TARGETS.get(competition_name, (None, DEFAULT_DIRECTION))


def choose_starting_parameters(
    competition_name: str,
    task_prompt: str,
    config: ExperimentConfig,
    controller: ParameterController,
    history: ParameterHistory,
) -> Tuple[float, float]:
    """Pick the episode's starting ``(temperature, top_p)``.

    Precedence: the best parameters a previous episode found for this task and
    model, then an explicit override from the config, then the default for the
    detected task family.

    Historical parameters are nudged slightly downwards before reuse. The
    controller's update rule raises the parameters after a success, so starting
    exactly at the recorded optimum would push the search off it immediately;
    starting just below leaves room to converge back onto it.
    """
    historical_temperature, historical_top_p = history.best_parameters(
        competition_name, config.generator.model
    )
    if historical_temperature is not None and historical_top_p is not None:
        temperature = max(TEMPERATURE_HARD_MIN, historical_temperature - HISTORICAL_TEMPERATURE_BACKOFF)
        top_p = max(TOP_P_HARD_MIN, historical_top_p - HISTORICAL_TOP_P_BACKOFF)
        LOGGER.info(
            "Starting from history: temperature=%.2f top_p=%.2f (recorded best %.2f / %.2f)",
            temperature, top_p, historical_temperature, historical_top_p,
        )
        return temperature, top_p

    if config.initial_temperature is not None and config.initial_top_p is not None:
        LOGGER.info(
            "Starting from config: temperature=%.2f top_p=%.2f",
            config.initial_temperature, config.initial_top_p,
        )
        return config.initial_temperature, config.initial_top_p

    problem_type = controller.detect_problem_type(competition_name, task_prompt)
    temperature, top_p = controller.initial_parameters(problem_type)
    LOGGER.info(
        "Detected task family '%s'; starting at temperature=%.2f top_p=%.2f",
        problem_type, temperature, top_p,
    )
    return temperature, top_p


class Episode:
    """Runs the attempt loop for one task."""

    def __init__(
        self,
        competition_name: str,
        config: ExperimentConfig,
        generator: GeneratorClient,
        corrector: CorrectorClient,
        param_history: ParameterHistory,
        recorder: RunRecorder,
        episode_tag: str,
        episode_start_time: str,
    ) -> None:
        self.competition_name = competition_name
        self.config = config
        self.generator = generator
        self.corrector = corrector
        self.param_history = param_history
        self.recorder = recorder
        self.episode_tag = episode_tag
        self.episode_start_time = episode_start_time

        self.workspace = EpisodeWorkspace(competition_name, episode_tag, config.output_root)
        self.controller = ParameterController(
            max_attempts=config.n_meta,
            config_adjustment=config.config_adjustment,
        )

        # Best result seen in this episode.
        self.best_performance: Optional[float] = None
        self.best_temperature: Optional[float] = None
        self.best_top_p: Optional[float] = None
        self.performance_based_selection = False
        self.best_metrics: Optional[Dict[str, str]] = None

    # -- generation --------------------------------------------------------- #

    def _generate_initial(
        self, task_prompt: str, temperature: float, top_p: float, attempt_num: int
    ) -> Tuple[Optional[str], str, bool]:
        """First attempt: generate straight from the task description."""
        response, tokens = self.generator.complete(
            generation_system_prompt(), task_prompt, temperature, top_p
        )
        self.recorder.add_inference_tokens(tokens)
        self.workspace.save_response("initial", attempt_num, response)

        code, from_answer_tag = extract_program(response)
        if code is not None:
            self.workspace.save_code("extracted", attempt_num, code)
        return code, response, from_answer_tag

    def _generate_optimized(
        self,
        task_prompt: str,
        reference_response: str,
        metrics: Optional[Dict[str, str]],
        temperature: float,
        top_p: float,
        attempt_num: int,
    ) -> Tuple[Optional[str], str, bool]:
        """Later attempt: improve on the previous working solution."""
        prompt = build_optimization_prompt(
            self.competition_name, task_prompt, reference_response, metrics
        )
        response, tokens = self.generator.complete(
            optimization_system_prompt(), prompt, temperature, top_p
        )
        self.recorder.add_inference_tokens(tokens)
        self.workspace.save_response("optimized", attempt_num, response)

        code, from_answer_tag = extract_program(response)
        if code is not None:
            self.workspace.save_code("optimized_extracted", attempt_num, code)
        else:
            LOGGER.warning("Attempt %d: no program in the optimisation response", attempt_num)
        return code, response, from_answer_tag

    def _generate_regenerated(
        self,
        task_prompt: str,
        previous_result: Optional[Dict[str, Any]],
        regeneration_reason: Optional[str],
        temperature: float,
        top_p: float,
        attempt_num: int,
    ) -> Tuple[Optional[str], str, bool]:
        """Later attempt: start over, informed by the errors seen so far.

        When the previous attempt exhausted its repair budget, the accumulated
        error messages are first summarised by the auxiliary model and folded
        into the task description, so the policy model is told what not to do
        again instead of being asked the same question twice.
        """
        prompt = None
        if previous_result and regeneration_reason == "correction_failed":
            summary = summarise_errors(self.corrector, previous_result.get("error_messages", []))
            if summary is not None:
                analysis, tokens = summary
                self.recorder.add_correction_tokens(tokens)
                prompt = build_error_informed_task_prompt(
                    self.competition_name, task_prompt, analysis
                )

        if prompt is None:
            prompt = build_regeneration_prompt(
                self.competition_name, task_prompt, regeneration_reason
            )

        response, tokens = self.generator.complete(
            generation_system_prompt(), prompt, temperature, top_p
        )
        self.recorder.add_inference_tokens(tokens)
        self.workspace.save_response("regenerated", attempt_num, response)

        code, from_answer_tag = extract_program(response)
        if code is not None:
            self.workspace.save_code("regenerated_extracted", attempt_num, code)
        else:
            LOGGER.warning("Attempt %d: no program in the regeneration response", attempt_num)
        return code, response, from_answer_tag

    def _current_gap(self) -> Optional[float]:
        """``S_gap`` -- normalised distance from the target metric."""
        if self.best_performance is None:
            return None
        gap, _met = self.controller.score_error_ratio(self.best_performance)
        return gap

    @staticmethod
    def _regeneration_reason(history: List[Dict[str, Any]]) -> Optional[str]:
        """Machine-readable trigger passed to the regeneration prompt builder."""
        if not history:
            return None
        return "extract_failed" if not history[-1]["code_extracted"] else "correction_failed"

    def _local_repair(
        self, previous_code: str, previous_error: str, meta_step: int
    ) -> Tuple[Optional[str], str, bool]:
        """``LocalRepair(LastCode(H), LastFeedback(H))`` -- Algorithm 1, line 10.

        Used when SATE selects ``repair``: the meta-step starts from a patched
        version of the last program rather than from a new generation, so the
        policy model is not called at all. This is the branch that keeps a
        pipeline alive when its repair budget ran out on an error that was
        still changing.
        """
        prompt = build_generic_correction_prompt(
            previous_code, previous_error, self.competition_name
        )
        response, tokens = self.corrector.complete(
            prompt=prompt,
            system_message=code_correction_system_prompt(self.competition_name),
            mode="correction",
        )
        self.recorder.add_correction_tokens(tokens)
        self.workspace.save_response("local_repair", meta_step, response)

        code = extract_fenced_code(response)
        if code and len(code.strip()) >= 10:
            self.workspace.save_code("local_repair_extracted", meta_step, code)
            return code, response, True
        LOGGER.warning("Meta-step %d: LocalRepair returned no usable program", meta_step)
        return None, response, False

    # -- submission --------------------------------------------------------- #

    def _submit_and_score(
        self, attempt_num: int, temperature: float, top_p: float
    ) -> Tuple[bool, Optional[Dict[str, str]], Optional[str]]:
        """Submit the produced file and read its score back.

        Returns ``(submitted, metrics, message)``. ``submitted`` is False when no
        submission file could be found, which the caller treats as an execution
        failure even though the process exited cleanly.
        """
        submission_path = find_submission_file(self.competition_name, self.config.output_root)
        if submission_path is None:
            LOGGER.warning("Attempt %d ran successfully but wrote no submission file", attempt_num)
            return False, None, None

        message = (
            f"node={self.config.node_id} episode={self.episode_tag} attempt={attempt_num} "
            f"model={self.config.generator.model} temp={temperature:.2f} top_p={top_p:.2f} "
            f"time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if not self.config.submit_to_leaderboard:
            LOGGER.info("Leaderboard submission disabled; keeping %s", submission_path)
            self.workspace.archive_submission(
                submission_path, attempt_num, datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            return True, None, message

        output = submit(self.competition_name, submission_path, message)
        LOGGER.info("Attempt %d submission output: %s", attempt_num, output.strip()[:300])

        self.workspace.archive_submission(
            submission_path, attempt_num, datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        if "failed" in output.lower():
            return True, None, message

        LOGGER.info("Waiting %d s for the leaderboard to score the submission", SUBMISSION_POLL_DELAY_SEC)
        time.sleep(SUBMISSION_POLL_DELAY_SEC)
        metrics = parse_metrics(latest_submission_row(self.competition_name))
        return True, metrics, message

    def _update_best(
        self, metrics: Optional[Dict[str, str]], temperature: float, top_p: float
    ) -> None:
        """Track the best private score seen in this episode."""
        if not metrics:
            return
        private_score = metrics.get("private_score")
        if not private_score or private_score == UNKNOWN_SCORE:
            return

        score = float(private_score)
        if self.best_performance is None or is_better(
            score, self.best_performance, self.competition_name
        ):
            self.best_performance = score
            self.best_temperature = temperature
            self.best_top_p = top_p
            self.performance_based_selection = True
            self.best_metrics = metrics
            LOGGER.info(
                "New best score %.6f at temperature=%.2f top_p=%.2f", score, temperature, top_p
            )

    # -- main loop ---------------------------------------------------------- #

    def run(self) -> EpisodeResult:
        """Execute the attempt loop and return the episode's contribution."""
        task_prompt = load_task_prompt(self.config.prompts_path, self.competition_name)
        target, direction = resolve_target(self.competition_name)
        self.controller.set_target(target, direction)

        temperature, top_p = choose_starting_parameters(
            self.competition_name, task_prompt, self.config, self.controller, self.param_history
        )
        self.controller.set_search_window(temperature, top_p)

        # The reasoning trace fed to the optimisation prompt is the one from the
        # first attempt, deliberately: it is the only trace produced from the
        # untouched task description, and reusing it keeps the optimisation
        # prompt identical in structure across attempts.
        initial_response = ""
        metrics: Optional[Dict[str, str]] = None

        meta_history: List[Dict[str, Any]] = []       # H in Algorithm 1
        initial_response = ""
        metrics: Optional[Dict[str, str]] = None

        for meta_step in range(1, self.config.n_meta + 1):
            # -- 1. state and trajectory action (Algorithm 1, lines 3-8) ------ #
            if meta_step == 1:
                action = "generate"
                LOGGER.info(
                    "=== %s | meta-step %d/%d | a_t=generate | theta_0=(%.2f, %.2f) ===",
                    self.competition_name, meta_step, self.config.n_meta, temperature, top_p,
                )
            else:
                state = build_state(
                    history=meta_history,
                    gap=self._current_gap(),
                    n_repair=self.config.n_repair,
                )
                action = select_action(state)
                LOGGER.info(
                    "=== %s | meta-step %d/%d | a_t=%s | theta_t=(%.2f, %.2f) ===",
                    self.competition_name, meta_step, self.config.n_meta, action,
                    temperature, top_p,
                )

            self.recorder.start_attempt(meta_step, temperature, top_p)

            generation_success = False
            execution_success = False
            from_answer_tag = True
            regeneration_reason: Optional[str] = None
            error_message = ""
            result: Optional[Dict[str, Any]] = None
            metrics = None

            # -- 2. produce C_t,0 (Algorithm 1, lines 9-13) -------------------- #
            if action == "generate":
                code, response, from_answer_tag = self._generate_initial(
                    task_prompt, temperature, top_p, meta_step
                )
                initial_response = response
            elif action == "repair":
                last = meta_history[-1]
                code, _response, from_answer_tag = self._local_repair(
                    last["code"], last.get("last_error", ""), meta_step
                )
            elif action == "refine":
                code, _response, from_answer_tag = self._generate_optimized(
                    task_prompt, initial_response, self.best_metrics, temperature, top_p, meta_step
                )
            else:  # regenerate
                code, _response, from_answer_tag = self._generate_regenerated(
                    task_prompt,
                    meta_history[-1].get("result") if meta_history else None,
                    self._regeneration_reason(meta_history),
                    temperature, top_p, meta_step,
                )

            # -- 3. execution-guided refinement loop (lines 14-41) ------------- #
            if code is None:
                regeneration_reason = "extract_failed"
                error_message = "no executable program could be extracted"
                self.recorder.finish_attempt("extract_failed")
            else:
                generation_success = True
                result = self_correct_and_execute(
                    code,
                    self.competition_name,
                    self.corrector,
                    self.config.execution,
                    self.recorder,
                    n_repair=self.config.n_repair,
                )
                self.workspace.save_code("corrected", meta_step, result["current_code"])

                execution_success = result.get("status") == "completed" or result.get(
                    "final_success", False
                )

                if execution_success:
                    submitted, metrics, message = self._submit_and_score(
                        meta_step, temperature, top_p
                    )
                    if submitted:
                        self.recorder.finish_attempt("execution_success", message, metrics)
                        self._update_best(metrics, temperature, top_p)
                    else:
                        execution_success = False
                        regeneration_reason = "correction_failed"
                        error_message = "the program ran but wrote no submission file"
                        self.recorder.finish_attempt("execution_failed")
                else:
                    regeneration_reason = "correction_failed"
                    if result.get("error_messages"):
                        error_message = result["error_messages"][-1]
                    self.recorder.finish_attempt("execution_failed")

            self.recorder.save_attempt(self.competition_name)

            # -- 4. append to H (lines 26 / 34 / 37) --------------------------- #
            meta_history.append(
                {
                    "meta_step": meta_step,
                    "action": action,
                    "code": result["current_code"] if result else (code or ""),
                    "code_extracted": code is not None,
                    "execution_success": execution_success,
                    "correction_count": result.get("correction_count", 0) if result else 0,
                    "deadlock": result.get("deadlock", False) if result else False,
                    "timed_out": result.get("termination") == "timeout" if result else False,
                    "error_messages": result.get("error_messages", []) if result else [],
                    "last_error": error_message,
                    "metrics": metrics,
                    "result": result,
                }
            )

            # -- 5. score the meta-step and pick theta_{t+1} ------------------- #
            new_temperature, new_top_p, reason, target_met = self.controller.update(
                temperature=temperature,
                top_p=top_p,
                generation_success=generation_success,
                regeneration_reason=regeneration_reason,
                execution_success=execution_success,
                error_message=error_message,
                attempt_num=meta_step,
                result=result,
                current_performance=metrics.get("private_score") if metrics else None,
                from_answer_tag=from_answer_tag,
            )
            self._log_reward(meta_step)

            # MeetTarget(P_t,k, P_target, D) -> return C_t,k (line 27-29)
            if target_met:
                LOGGER.info(
                    "MeetTarget satisfied at meta-step %d; returning C*", meta_step
                )
                break

            if meta_step < self.config.n_meta:
                LOGGER.info("theta update: %s", reason)
                LOGGER.info(
                    "temperature %.2f -> %.2f, top_p %.2f -> %.2f",
                    temperature, new_temperature, top_p, new_top_p,
                )
                temperature, top_p = new_temperature, new_top_p

        self.meta_history = meta_history  # H, kept for inspection after run()
        return self._finalise()

    def _log_reward(self, attempt_num: int) -> None:
        """Log the reward decomposition of the most recent attempt."""
        if not self.controller.reward_history:
            return
        reward = self.controller.reward_history[-1]
        LOGGER.info(
            "Attempt %d reward %.1f | generation=%s execution=%s efficiency=%.1f "
            "performance=%.1f (recorded only)",
            attempt_num,
            reward["total_reward"],
            reward["r_generation"],
            reward["r_execution"],
            reward["r_efficiency"],
            reward["r_performance"],
        )

    def _finalise(self) -> EpisodeResult:
        """Select the episode's best configuration and write the parameter log."""
        if not self.performance_based_selection and self.best_temperature is None:
            temperature, top_p, best_reward = self.controller.best_by_reward()
            if temperature is not None:
                self.best_temperature, self.best_top_p = temperature, top_p
                LOGGER.info(
                    "No leaderboard score available; selected by reward: "
                    "temperature=%.2f top_p=%.2f reward=%.1f",
                    temperature, top_p, best_reward,
                )

        actual_attempts = len(self.controller.param_history)
        selection_method = "performance" if self.performance_based_selection else "reward"

        if self.controller.param_history:
            self.workspace.save_parameter_log(
                {
                    "episode_start_time": self.episode_start_time,
                    "episode_tag": self.episode_tag,
                    "competition_name": self.competition_name,
                    "n_meta": self.config.n_meta,
                    "n_repair": self.config.n_repair,
                    "actual_meta_steps": actual_attempts,
                    "attempt_history": [dict(record) for record in self.controller.param_history],
                    "reward_history": [dict(record) for record in self.controller.reward_history],
                    "failure_patterns": self.controller.failure_patterns,
                    "best_config": (
                        {
                            "temperature": self.best_temperature,
                            "top_p": self.best_top_p,
                            "performance": self.best_performance,
                            "selection_method": selection_method,
                        }
                        if self.best_temperature is not None
                        else None
                    ),
                }
            )

        LOGGER.info(
            "Trajectory summary for %s: %d/%d meta-steps, best score %s",
            self.competition_name, actual_attempts, self.config.n_meta, self.best_performance,
        )

        return EpisodeResult(
            best_temperature=self.best_temperature,
            best_top_p=self.best_top_p,
            best_performance=self.best_performance,
            actual_attempts=actual_attempts,
            selection_method=selection_method,
        )


def run_episode(
    competition_name: str,
    config: ExperimentConfig,
    generator: GeneratorClient,
    corrector: CorrectorClient,
    history: ParameterHistory,
    recorder: RunRecorder,
    episode_tag: str,
    episode_start_time: str,
) -> EpisodeResult:
    """Convenience wrapper around :class:`Episode`."""
    episode = Episode(
        competition_name=competition_name,
        config=config,
        generator=generator,
        corrector=corrector,
        param_history=history,
        recorder=recorder,
        episode_tag=episode_tag,
        episode_start_time=episode_start_time,
    )
    try:
        return episode.run()
    except Exception as exc:
        LOGGER.error("Episode failed for %s: %s", competition_name, exc)
        LOGGER.debug(traceback.format_exc())
        raise
