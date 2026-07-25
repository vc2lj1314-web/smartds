"""State-Aware Trajectory Exploration: state construction and action selection.

This module implements the two SATE responsibilities that Algorithm 1 names
directly:

* ``BuildState(H, D)`` -- Eq. 3, the state at meta-step *t*

      S_t = { S_gap, H_t, S_mode, S_sim }

* ``SATE(S_t, D, H)`` -- the trajectory action a_t in
  {refine, repair, regenerate}, together with the prompt Q_t it implies.

The third component the controller returns, the decoding configuration theta_t,
is produced by :mod:`autods.parameter_learning`; it is kept separate because
the decoding controller carries reward history that the action selector does
not need.

Action selection follows Sec. 3.2 and 3.3 of the paper:

===================================  ==================================  ===========
S_mode after a meta-step             reading                             a_t
===================================  ==================================  ===========
metric_underperformance              executable, below target            refine
localized_failure                    syntax/runtime error, errors still  repair
                                     changing, repair budget spent
semantic_deadlock                    same failure reproduced             regenerate
correction_exhaustion                budget spent, no localized handle   regenerate
resource_timeout                     approach too expensive              regenerate
extraction_failure                   no program recovered                regenerate
===================================  ==================================  ===========

The ``repair`` branch is what makes Algorithm 1 lines 9-13 reachable: when a
meta-step ends having spent its repair budget on an error that is still
*moving*, SATE grants a fresh repair budget in the next meta-step instead of
discarding a pipeline that may be one fix away from running. The policy model
is not called on that meta-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .config import TAU_SIM
from .logging_utils import get_logger
from .similarity import max_pairwise_similarity

LOGGER = get_logger(__name__)

TrajectoryAction = Literal["generate", "refine", "repair", "regenerate"]

FailureMode = Literal[
    "metric_underperformance",
    "localized_failure",
    "semantic_deadlock",
    "correction_exhaustion",
    "resource_timeout",
    "extraction_failure",
]


@dataclass
class MetaState:
    """S_t as defined in Eq. 3.

    Attributes:
        gap: ``S_gap`` -- normalised distance from the target metric. ``None``
            when no score is available, which is itself informative: it means
            no candidate has executed yet.
        history: ``H_t`` -- ordered record of previous meta-steps, each holding
            the code, the execution outcome and the observed metric.
        mode: ``S_mode`` -- how the previous meta-step ended.
        sim: ``S_sim`` -- max semantic similarity among the error logs of the
            previous meta-step.
    """

    gap: Optional[float]
    history: List[Dict[str, Any]] = field(default_factory=list)
    mode: FailureMode = "extraction_failure"
    sim: float = 0.0

    def describe(self) -> str:
        gap_text = "n/a" if self.gap is None else f"{self.gap:+.4f}"
        return (
            f"S_gap={gap_text} |H_t|={len(self.history)} "
            f"S_mode={self.mode} S_sim={self.sim:.3f}"
        )


def classify_mode(
    outcome: Dict[str, Any],
    n_repair: int,
    tau_sim: float = TAU_SIM,
) -> FailureMode:
    """Derive ``S_mode`` from how a meta-step's refinement loop terminated.

    Args:
        outcome: The finished correction-loop state.
        n_repair: The repair budget the loop was given.
        tau_sim: Deadlock threshold.
    """
    if not outcome.get("code_extracted", True):
        return "extraction_failure"

    if outcome.get("execution_success"):
        return "metric_underperformance"

    if outcome.get("deadlock"):
        return "semantic_deadlock"

    if outcome.get("timed_out"):
        return "resource_timeout"

    errors: List[str] = outcome.get("error_messages", [])
    if outcome.get("correction_count", 0) >= n_repair:
        # The budget is spent. Whether the pipeline is worth another repair
        # budget depends on whether the error was still moving when the budget
        # ran out; a stationary error is a deadlock in all but name.
        if errors and max_pairwise_similarity(errors) >= tau_sim:
            return "semantic_deadlock"
        return "localized_failure"

    return "correction_exhaustion"


def build_state(
    history: List[Dict[str, Any]],
    gap: Optional[float],
    n_repair: int,
    tau_sim: float = TAU_SIM,
) -> MetaState:
    """``BuildState(H, D)`` -- assemble S_t from the accumulated history."""
    if not history:
        return MetaState(gap=gap, history=history, mode="extraction_failure", sim=0.0)

    last = history[-1]
    errors: List[str] = last.get("error_messages", [])
    return MetaState(
        gap=gap,
        history=history,
        mode=classify_mode(last, n_repair, tau_sim),
        sim=max_pairwise_similarity(errors) if errors else 0.0,
    )


#: S_mode -> a_t. Kept as data so the policy can be read off in one place and
#: cited against the table in the module docstring.
ACTION_BY_MODE: Dict[FailureMode, TrajectoryAction] = {
    "metric_underperformance": "refine",
    "localized_failure": "repair",
    "semantic_deadlock": "regenerate",
    "correction_exhaustion": "regenerate",
    "resource_timeout": "regenerate",
    "extraction_failure": "regenerate",
}


def select_action(state: MetaState) -> TrajectoryAction:
    """``SATE(S_t, D, H)`` -- pick the trajectory action for the next meta-step."""
    action = ACTION_BY_MODE.get(state.mode, "regenerate")
    LOGGER.info("SATE: %s -> a_t=%s", state.describe(), action)
    return action
