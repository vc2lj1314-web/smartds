"""The adaptive sampling-parameter controller.

One episode gives the policy model a small number of attempts at a task. After
each attempt the controller receives a reward decomposed into four terms and
moves the sampling temperature and top-p, jointly, in a direction that depends
on *how* the attempt failed rather than only on whether it failed:

* a response the code extractor could not parse means the sampling is too
  conservative, so both parameters go **up**;
* a program that ran but produced a poor score means the sampling can afford to
  be more exploratory, so both parameters go **up**, scaled by how far the score
  is from the target;
* a program that crashed means the sampling is too exploratory, so both
  parameters go **down**, scaled by how severe the crash was.

Two mechanisms keep the search stable. Step sizes shrink as a parameter
approaches the edge of its window (:meth:`adaptive_step`), so the search does
not slam into a boundary and stay there. And the temperature may not be pushed
below a value that has already been shown to execute successfully in this
episode (:meth:`successful_temperature_floor`).

Reward decomposition
--------------------
``total = alpha * r_generation + beta * r_execution + delta * r_efficiency``

``r_performance`` is computed and logged but carries weight zero. It is
deliberately excluded from the control signal: it is the leaderboard score, and
letting it steer the sampling parameters within an episode would tune the
harness against the evaluation metric. It is retained in the record because it
is used for post-hoc analysis and for selecting the episode's best parameters.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

from .code_analysis import is_timeout_error
from .config import (
    CONFIG_ADJUSTMENT_TEMPERATURE_STEP,
    CONFIG_ADJUSTMENT_TOP_P_STEP,
    TAU_SIM,
    N_REPAIR,
    PROBLEM_TYPE_BASE_CONFIGS,
    PROBLEM_TYPE_KEYWORDS,
    TEMPERATURE_HARD_MAX,
    TEMPERATURE_HARD_MIN,
    TEMPERATURE_WINDOW_DOWN,
    TEMPERATURE_WINDOW_UP,
    TOP_P_HARD_MAX,
    TOP_P_HARD_MIN,
    TOP_P_WINDOW_DOWN,
    TOP_P_WINDOW_UP,
)
from .logging_utils import get_logger
from .similarity import max_pairwise_similarity

LOGGER = get_logger(__name__)

HISTORY_LENGTH = 20

# -- reward constants ------------------------------------------------------- #

R_GEN_ANSWER_TAG = 50      # program recovered from the <answer> contract
R_GEN_FALLBACK = 35        # program recovered only by the fallback scan
R_GEN_FAILED = -100        # no program could be recovered

#: Execution reward by number of repairs needed; fewer repairs is better.
R_EXEC_BY_CORRECTIONS = {0: 150, 1: 120, 2: 90}
R_EXEC_MANY_CORRECTIONS = 60   # 3 or 4 repairs
R_EXEC_MAX_CORRECTIONS = 30    # 5 or more repairs

#: Execution penalty by failure severity, most severe first.
R_EXEC_PENALTY_BY_SEVERITY = {1: -300, 2: -250, 3: -200}
R_EXEC_PENALTY_NO_PROGRAM = -200
R_EXEC_PENALTY_OTHER = -150

R_EFFICIENCY_PER_REMAINING_ATTEMPT = 10
R_PERF_TARGET_MET_BASE = 100
R_PERF_FIRST_ATTEMPT_BONUS = 30

#: Reward weights, annealed over the attempts of an episode: early attempts
#: weight format compliance most, later ones weight execution.
ALPHA_START, ALPHA_DECAY, ALPHA_FLOOR = 0.5, 0.1, 0.2
BETA_START, BETA_DECAY, BETA_FLOOR = 0.4, 0.05, 0.2
DELTA = 0.1

#: Base step sizes.
FIRST_ATTEMPT_TEMPERATURE_STEP = 0.08
FIRST_ATTEMPT_TOP_P_STEP = 0.02
LATER_ATTEMPT_TEMPERATURE_STEP = 0.06
LATER_ATTEMPT_TOP_P_STEP = 0.02

#: Escalating steps applied after repeated extraction failures.
EXTRACT_FAILURE_STEPS = (
    (0.08, 0.03, "first extraction failure: small increase in sampling diversity"),
    (0.12, 0.05, "second extraction failure: moderate increase in sampling diversity"),
    (0.18, 0.08, "repeated extraction failure: large increase in sampling diversity"),
)

#: Reward-change scaling. The raw reward difference is on the order of hundreds,
#: so it is scaled down hard before it is allowed to move a parameter.
REWARD_CHANGE_SCALE = 0.01
REWARD_CHANGE_CLIP = 5.0
REWARD_CHANGE_ACTIVATION = 0.5
REWARD_TO_TEMPERATURE = 0.02
REWARD_TO_TOP_P = 0.008
CONFIDENCE_SATURATION = 5.0

#: Fraction of a step that is removed as a parameter approaches its boundary.
BOUNDARY_DAMPING = 0.7

#: Failure taxonomy: label and step multiplier. A more severe failure justifies
#: a larger corrective move.
FAILURE_SEVERITY = (
    ("critical_failure", 2.5, ("segmentation fault", "core dumped", "fatal", "killed -9")),
    ("severe_failure", 2.0, ("timeout", "memory", "killed", "cuda out of memory", "process killed")),
    ("moderate_failure", 1.5, ("runtime", "runtimeerror", "filenotfound", "permission",
                               "valueerror", "typeerror", "indexerror", "keyerror")),
    ("mild_failure", 0.8, ("syntax", "import", "syntaxerror", "importerror",
                           "modulenotfounderror", "nameerror")),
    ("minor_failure", 0.5, ("warning", "deprecated", "futurewarning")),
)
DEFAULT_FAILURE_SEVERITY = ("general_failure", 1.2)

#: Non-linear mapping from relative score error to step multiplier: a larger gap
#: to the target justifies a larger move, with diminishing returns.
PERF_FACTOR_SMALL_ERROR = 0.1
PERF_FACTOR_MEDIUM_ERROR = 0.5


class ParameterController:
    """Adaptive controller over ``(temperature, top_p)`` for one episode."""

    def __init__(
        self,
        max_attempts: int = 3,
        config_adjustment: int = 0,
    ) -> None:
        """Create a controller.

        Args:
            max_attempts: N_meta, the meta-step budget of the trajectory; used
                by the efficiency reward term.
            config_adjustment: Integer offset applied to every starting point,
                so that several nodes exploring the same task in parallel do not
                all start from the same place.
        """
        self.max_attempts = max_attempts
        self.param_history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_LENGTH)
        self.reward_history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_LENGTH)
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}

        self.target_performance: Optional[float] = None
        self.problem_direction: Optional[str] = None

        temp_offset = CONFIG_ADJUSTMENT_TEMPERATURE_STEP * config_adjustment
        top_p_offset = CONFIG_ADJUSTMENT_TOP_P_STEP * config_adjustment
        self.problem_type_configs = {
            name: (temp + temp_offset, top_p + top_p_offset)
            for name, (temp, top_p) in PROBLEM_TYPE_BASE_CONFIGS.items()
        }

        self.extract_failed_count = 0
        self.temperature_window: Optional[Dict[str, float]] = None
        self.top_p_window: Optional[Dict[str, float]] = None

    # -- set-up ------------------------------------------------------------- #

    def set_target(self, target_score: Optional[float], direction: str = "maximize") -> None:
        """Record the task's target score and optimisation direction."""
        self.target_performance = target_score
        self.problem_direction = direction
        LOGGER.info("Target score %s (%s)", target_score, direction)

    def set_search_window(self, initial_temperature: float, initial_top_p: float) -> None:
        """Derive the per-episode search window from the starting point.

        The window is centred on where the episode starts and then clipped to
        the global hard bounds, so an episode that starts from a high historical
        temperature does not get a wider absolute range than one starting low.
        """
        self.temperature_window = {
            "min": max(TEMPERATURE_HARD_MIN, initial_temperature - TEMPERATURE_WINDOW_DOWN),
            "max": min(TEMPERATURE_HARD_MAX, initial_temperature + TEMPERATURE_WINDOW_UP),
            "initial": initial_temperature,
        }
        self.top_p_window = {
            "min": max(TOP_P_HARD_MIN, initial_top_p - TOP_P_WINDOW_DOWN),
            "max": min(TOP_P_HARD_MAX, initial_top_p + TOP_P_WINDOW_UP),
            "initial": initial_top_p,
        }
        LOGGER.info(
            "Search window: temperature [%.2f, %.2f], top_p [%.2f, %.2f]",
            self.temperature_window["min"],
            self.temperature_window["max"],
            self.top_p_window["min"],
            self.top_p_window["max"],
        )

    def initial_parameters(self, problem_type: str = "default") -> Tuple[float, float]:
        """Starting ``(temperature, top_p)`` for a task family."""
        return self.problem_type_configs.get(problem_type, self.problem_type_configs["default"])

    @staticmethod
    def detect_problem_type(competition_name: str, prompt_content: str = "") -> str:
        """Guess the task family from the task name and its prompt.

        Falls back to ``tabular_data``, which is both the most common family in
        the benchmark and the most conservative starting point.
        """
        haystack = f"{competition_name} {prompt_content}".lower()
        for problem_type, keywords in PROBLEM_TYPE_KEYWORDS.items():
            if any(keyword in haystack for keyword in keywords):
                return problem_type
        return "tabular_data"

    # -- diagnosis ---------------------------------------------------------- #

    @staticmethod
    def classify_failure(error_message: Optional[str]) -> Tuple[str, float]:
        """Map an error message to ``(label, step_multiplier)``."""
        if not error_message:
            return "unknown", 1.0

        lowered = error_message.lower()
        for label, multiplier, keywords in FAILURE_SEVERITY:
            if any(keyword in lowered for keyword in keywords):
                return label, multiplier
        return DEFAULT_FAILURE_SEVERITY

    @staticmethod
    def classify_execution_outcome(result: Optional[Dict[str, Any]]) -> Tuple[str, int]:
        """Classify why a correction loop ended without a submission.

        Returns ``(label, severity)`` with severity 1 the most serious. The
        ordering reflects how much of the budget was wasted: an unchanging error
        means every repair was useless, a timeout means the approach is too
        expensive, and exhausting the repair budget is the ordinary case.
        """
        if not result:
            return "general_failure", 3

        error_messages: List[str] = result.get("error_messages", [])
        status = result.get("status", "")
        correction_count = result.get("correction_count", 0)

        if len(error_messages) >= 2 and max_pairwise_similarity(error_messages) >= TAU_SIM:
            return "similarity_failure", 1

        if status == "timeout_error" or any(is_timeout_error(message) for message in error_messages):
            return "timeout_failure", 2

        if correction_count >= N_REPAIR:
            return "correction_limit_failure", 3

        return "general_failure", 3

    def score_error_ratio(self, current_performance: Any) -> Tuple[Optional[float], bool]:
        """Relative distance from the target score, signed so that >= 0 means met.

        Returns ``(ratio, target_met)``, or ``(None, False)`` when no target or
        no score is available.
        """
        if self.target_performance is None or current_performance is None:
            return None, False

        try:
            current = float(current_performance)
            target = float(self.target_performance)
        except (TypeError, ValueError):
            return None, False

        if target == 0:
            return None, False

        if self.problem_direction == "maximize":
            ratio = (current - target) / target
        else:
            ratio = (target - current) / target

        target_met = ratio >= 0
        LOGGER.info(
            "Score check: current=%.6f target=%.6f relative error=%.4f met=%s",
            current, target, ratio, target_met,
        )
        return ratio, target_met

    @staticmethod
    def performance_step_factor(error_ratio: Optional[float]) -> float:
        """Turn a relative score error into a step multiplier.

        Piecewise: linear for small errors, gentler above 10 %, logarithmic
        above 50 % so that a task the model is nowhere near solving does not
        produce an unbounded jump.
        """
        if error_ratio is None:
            return 1.0

        magnitude = abs(error_ratio)
        if magnitude <= PERF_FACTOR_SMALL_ERROR:
            return 1.0 + magnitude * 2
        if magnitude <= PERF_FACTOR_MEDIUM_ERROR:
            return 1.2 + (magnitude - PERF_FACTOR_SMALL_ERROR) * 1.0
        return 1.6 + min(math.log(magnitude), 1.0)

    def extract_failure_step(self) -> Tuple[float, float, str]:
        """Escalating step sizes for consecutive code-extraction failures."""
        self.extract_failed_count += 1
        index = min(self.extract_failed_count, len(EXTRACT_FAILURE_STEPS)) - 1
        temp_step, top_p_step, reason = EXTRACT_FAILURE_STEPS[index]
        if self.extract_failed_count > len(EXTRACT_FAILURE_STEPS):
            reason = f"{reason} (failure {self.extract_failed_count})"
        return temp_step, top_p_step, reason

    def reward_change_step(self) -> Tuple[float, float]:
        """Scaled change in total reward between the last two attempts.

        Returns ``(scaled_change, confidence)``; confidence grows with the
        amount of history available and saturates at 1.
        """
        if len(self.reward_history) < 2:
            return 0.0, 0.0

        recent = list(self.reward_history)[-2:]
        change = recent[-1]["total_reward"] - recent[-2]["total_reward"]
        scaled = max(-REWARD_CHANGE_CLIP, min(REWARD_CHANGE_CLIP, change * REWARD_CHANGE_SCALE))
        confidence = min(len(self.reward_history) / CONFIDENCE_SATURATION, 1.0)

        LOGGER.debug("Reward change %.4f -> scaled step %.4f", change, scaled)
        return scaled, confidence

    def adaptive_step(
        self,
        current_temperature: float,
        current_top_p: float,
        temperature_step: float,
        top_p_step: float,
        direction: str = "up",
    ) -> Tuple[float, float]:
        """Damp a step as the parameter approaches the edge of its window.

        A parameter already near the top of its window gets at most
        ``1 - BOUNDARY_DAMPING`` of an upward step, and symmetrically for
        downward steps near the bottom.
        """
        if self.temperature_window is None or self.top_p_window is None:
            return temperature_step, top_p_step

        def position(value: float, window: Dict[str, float]) -> float:
            span = window["max"] - window["min"]
            if span <= 0:
                return 0.5
            return max(0.0, min(1.0, (value - window["min"]) / span))

        temp_position = position(current_temperature, self.temperature_window)
        top_p_position = position(current_top_p, self.top_p_window)

        if direction == "up":
            temp_factor = 1.0 - temp_position * BOUNDARY_DAMPING
            top_p_factor = 1.0 - top_p_position * BOUNDARY_DAMPING
        else:
            temp_factor = 1.0 - (1.0 - temp_position) * BOUNDARY_DAMPING
            top_p_factor = 1.0 - (1.0 - top_p_position) * BOUNDARY_DAMPING

        return temperature_step * temp_factor, top_p_step * top_p_factor

    def successful_temperature_floor(self) -> float:
        """Lowest temperature the next attempt may use.

        Once a temperature has produced an executable program in this episode,
        the controller will not go below it: doing so would discard evidence
        that the model can succeed there. The history excludes the current
        attempt, which is the one being reacted to.
        """
        records = list(self.param_history)[:-1]
        if not records:
            return TEMPERATURE_HARD_MIN

        successful = [r["temperature"] for r in records if r.get("execution_success")]
        if not successful:
            return TEMPERATURE_HARD_MIN

        last = records[-1]
        if last.get("execution_success"):
            LOGGER.info(
                "Previous attempt succeeded and this one did not; floor set to %.2f",
                last["temperature"],
            )
            return last["temperature"]

        if len(successful) >= 2:
            floor = max(successful)
            LOGGER.info("%d successes on record; floor set to %.2f", len(successful), floor)
            return floor

        return successful[0]

    # -- main update -------------------------------------------------------- #

    def update(
        self,
        temperature: float,
        top_p: float,
        generation_success: bool,
        regeneration_reason: Optional[str],
        execution_success: bool,
        error_message: Optional[str],
        attempt_num: int,
        result: Optional[Dict[str, Any]] = None,
        current_performance: Any = None,
        from_answer_tag: bool = True,
    ) -> Tuple[float, float, str, bool]:
        """Score one attempt and produce the parameters for the next one.

        Args:
            temperature: Temperature used for the attempt being scored.
            top_p: Top-p used for the attempt being scored.
            generation_success: Whether a program could be extracted at all.
            regeneration_reason: ``"extract_failed"``, ``"correction_failed"``
                or ``None``.
            execution_success: Whether a submission was produced.
            error_message: Last error observed, if any.
            attempt_num: 1-based index of the attempt within the episode.
            result: Final state of the correction loop, if it ran.
            current_performance: Leaderboard score, if one came back.
            from_answer_tag: Whether the program came from the ``<answer>``
                contract rather than from the fallback scan.

        Returns:
            ``(next_temperature, next_top_p, reason, target_met)``. When
            ``target_met`` is true the caller should stop early: the task's
            reference score has already been reached.
        """
        reward = self._compute_reward(
            generation_success,
            execution_success,
            from_answer_tag,
            attempt_num,
            result,
            current_performance,
        )

        self.param_history.append(
            {
                "temperature": temperature,
                "top_p": top_p,
                "generation_success": generation_success,
                "execution_success": execution_success,
                "attempt_num": attempt_num,
                "regeneration_reason": regeneration_reason,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.reward_history.append(reward)
        self._record_failure_pattern(temperature, top_p, execution_success, error_message)

        if execution_success and current_performance is not None:
            _ratio, target_met = self.score_error_ratio(current_performance)
            if target_met:
                LOGGER.info("Target score reached; the episode can stop early")
                return temperature, top_p, "target score reached, no adjustment needed", True

        if attempt_num == 1:
            new_temperature, new_top_p, reason = self._first_attempt_update(
                temperature, top_p, execution_success, regeneration_reason
            )
        else:
            new_temperature, new_top_p, reason = self._later_attempt_update(
                temperature,
                top_p,
                execution_success,
                regeneration_reason,
                error_message,
                current_performance,
            )

        return new_temperature, new_top_p, reason, False

    # -- reward ------------------------------------------------------------- #

    def _compute_reward(
        self,
        generation_success: bool,
        execution_success: bool,
        from_answer_tag: bool,
        attempt_num: int,
        result: Optional[Dict[str, Any]],
        current_performance: Any,
    ) -> Dict[str, Any]:
        """Decompose the outcome of one attempt into reward terms."""
        if generation_success:
            r_generation = R_GEN_ANSWER_TAG if from_answer_tag else R_GEN_FALLBACK
        else:
            r_generation = R_GEN_FAILED

        if execution_success:
            correction_count = result.get("correction_count", 0) if result else 0
            if correction_count in R_EXEC_BY_CORRECTIONS:
                r_execution = R_EXEC_BY_CORRECTIONS[correction_count]
            elif correction_count <= 4:
                r_execution = R_EXEC_MANY_CORRECTIONS
            else:
                r_execution = R_EXEC_MAX_CORRECTIONS
        elif generation_success:
            _label, severity = self.classify_execution_outcome(result)
            r_execution = R_EXEC_PENALTY_BY_SEVERITY.get(severity, R_EXEC_PENALTY_OTHER)
        else:
            r_execution = R_EXEC_PENALTY_NO_PROGRAM

        r_efficiency = (self.max_attempts - attempt_num + 1) * R_EFFICIENCY_PER_REMAINING_ATTEMPT

        # Recorded but not used to steer the controller; see the module docstring.
        r_performance = 0.0
        if execution_success and current_performance is not None:
            error_ratio, target_met = self.score_error_ratio(current_performance)
            if error_ratio is None:
                r_performance = R_PERF_FIRST_ATTEMPT_BONUS if attempt_num == 1 else 0.0
            elif target_met:
                r_performance = R_PERF_TARGET_MET_BASE + min(error_ratio * 50, 50)
            else:
                r_performance = R_PERF_TARGET_MET_BASE - min(abs(error_ratio) * 50, 50)

        alpha = max(ALPHA_START - ALPHA_DECAY * (attempt_num - 1), ALPHA_FLOOR)
        beta = max(BETA_START - BETA_DECAY * (attempt_num - 1), BETA_FLOOR)
        total = alpha * r_generation + beta * r_execution + DELTA * r_efficiency

        return {
            "total_reward": total,
            "r_generation": r_generation,
            "r_execution": r_execution,
            "r_performance": r_performance,
            "r_efficiency": r_efficiency,
            "weights": {"alpha": alpha, "beta": beta, "performance": 0.0, "delta": DELTA},
        }

    def _record_failure_pattern(
        self,
        temperature: float,
        top_p: float,
        execution_success: bool,
        error_message: Optional[str],
    ) -> None:
        """Accumulate a diagnostic index of recurring errors.

        Purely observational: it is written to the run record for analysis and
        never read back by the controller.
        """
        if execution_success or not error_message:
            return

        key = error_message[:100]
        entry = self.failure_patterns.setdefault(
            key,
            {"count": 0, "params": [], "description": self.classify_failure(error_message)[0]},
        )
        entry["count"] += 1
        entry["params"].append((temperature, top_p))

    # -- parameter moves ----------------------------------------------------- #

    def _clip_up(self, temperature: float, top_p: float) -> Tuple[float, float]:
        temp_max = self.temperature_window["max"] if self.temperature_window else TEMPERATURE_HARD_MAX
        top_p_max = self.top_p_window["max"] if self.top_p_window else TOP_P_HARD_MAX
        return min(temperature, temp_max), min(top_p, top_p_max)

    def _clip_down(
        self, temperature: float, top_p: float, temperature_floor: Optional[float] = None
    ) -> Tuple[float, float]:
        temp_min = self.temperature_window["min"] if self.temperature_window else TEMPERATURE_HARD_MIN
        if temperature_floor is not None:
            temp_min = max(temp_min, temperature_floor)
        top_p_min = self.top_p_window["min"] if self.top_p_window else TOP_P_HARD_MIN
        return max(temperature, temp_min), max(top_p, top_p_min)

    def _first_attempt_update(
        self,
        temperature: float,
        top_p: float,
        execution_success: bool,
        regeneration_reason: Optional[str],
    ) -> Tuple[float, float, str]:
        """Fixed-size move after the first attempt, before any reward history exists."""
        if execution_success or regeneration_reason == "extract_failed":
            temp_step, top_p_step = self.adaptive_step(
                temperature, top_p,
                FIRST_ATTEMPT_TEMPERATURE_STEP, FIRST_ATTEMPT_TOP_P_STEP, "up",
            )
            new_temperature, new_top_p = self._clip_up(
                temperature + temp_step, top_p + top_p_step
            )
            trigger = "success" if execution_success else "extraction failure"
            reason = (
                f"first attempt ({trigger}): increasing sampling diversity "
                f"(temp +{temp_step:.3f}, top_p +{top_p_step:.3f})"
            )
        else:
            temp_step, top_p_step = self.adaptive_step(
                temperature, top_p,
                FIRST_ATTEMPT_TEMPERATURE_STEP, FIRST_ATTEMPT_TOP_P_STEP, "down",
            )
            new_temperature, new_top_p = self._clip_down(
                temperature - temp_step, top_p - top_p_step
            )
            reason = (
                f"first attempt (execution failure): increasing determinism "
                f"(temp -{temp_step:.3f}, top_p -{top_p_step:.3f})"
            )

        return new_temperature, new_top_p, reason

    def _later_attempt_update(
        self,
        temperature: float,
        top_p: float,
        execution_success: bool,
        regeneration_reason: Optional[str],
        error_message: Optional[str],
        current_performance: Any,
    ) -> Tuple[float, float, str]:
        """Reward- and diagnosis-driven move from the second attempt onwards."""
        reward_step, _confidence = self.reward_change_step()

        if regeneration_reason == "extract_failed":
            temp_step, top_p_step, escalation_reason = self.extract_failure_step()
            temp_step, top_p_step = self.adaptive_step(
                temperature, top_p, temp_step, top_p_step, "up"
            )
            new_temperature, new_top_p = self._clip_up(
                temperature + temp_step, top_p + top_p_step
            )
            reason = (
                f"{escalation_reason} (temp +{temp_step:.3f}, top_p +{top_p_step:.3f})"
            )
            return new_temperature, new_top_p, reason

        if execution_success:
            error_ratio, _ = self.score_error_ratio(current_performance)
            factor = self.performance_step_factor(error_ratio)

            if abs(reward_step) > REWARD_CHANGE_ACTIVATION:
                temp_step = abs(reward_step) * REWARD_TO_TEMPERATURE * factor
                top_p_step = abs(reward_step) * REWARD_TO_TOP_P * factor
                source = "reward change + score gap"
            else:
                temp_step = LATER_ATTEMPT_TEMPERATURE_STEP * factor
                top_p_step = LATER_ATTEMPT_TOP_P_STEP * factor
                source = "score gap"

            temp_step, top_p_step = self.adaptive_step(
                temperature, top_p, temp_step, top_p_step, "up"
            )
            new_temperature, new_top_p = self._clip_up(
                temperature + temp_step, top_p + top_p_step
            )
            reason = (
                f"execution succeeded, exploring upwards ({source}): factor={factor:.2f} "
                f"(temp +{temp_step:.3f}, top_p +{top_p_step:.3f})"
            )
            return new_temperature, new_top_p, reason

        failure_label, severity = self.classify_failure(error_message)
        severity_temp_step = LATER_ATTEMPT_TEMPERATURE_STEP * severity
        severity_top_p_step = LATER_ATTEMPT_TOP_P_STEP * severity

        if abs(reward_step) > REWARD_CHANGE_ACTIVATION:
            temp_step = max(abs(reward_step) * REWARD_TO_TEMPERATURE * severity, severity_temp_step)
            top_p_step = max(abs(reward_step) * REWARD_TO_TOP_P * severity, severity_top_p_step)
            source = "reward change + failure severity"
        else:
            temp_step, top_p_step = severity_temp_step, severity_top_p_step
            source = "failure severity"

        temp_step, top_p_step = self.adaptive_step(
            temperature, top_p, temp_step, top_p_step, "down"
        )
        new_temperature, new_top_p = self._clip_down(
            temperature - temp_step,
            top_p - top_p_step,
            temperature_floor=self.successful_temperature_floor(),
        )
        reason = (
            f"execution failed, backing off ({source}): failure={failure_label}, "
            f"severity={severity:.2f} (temp -{temp_step:.3f}, top_p -{top_p_step:.3f})"
        )
        return new_temperature, new_top_p, reason

    # -- selection ----------------------------------------------------------- #

    def best_by_reward(self) -> Tuple[Optional[float], Optional[float], float]:
        """Best executing parameters of this episode, judged by total reward.

        Used only when no leaderboard score came back at all, so that an episode
        still contributes something to the cross-episode history.
        """
        best_reward = -float("inf")
        best_temperature: Optional[float] = None
        best_top_p: Optional[float] = None

        for params, reward in zip(self.param_history, self.reward_history):
            if params.get("execution_success") and reward["total_reward"] > best_reward:
                best_reward = reward["total_reward"]
                best_temperature = params["temperature"]
                best_top_p = params["top_p"]

        return best_temperature, best_top_p, best_reward
