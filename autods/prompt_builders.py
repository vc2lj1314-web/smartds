"""Assembling the concrete prompts sent at each point of the pipeline.

Every builder here is a pure function of the templates in :mod:`autods.prompts`
and the run state, so the exact text sent to a model can be reconstructed from
the run record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .code_extraction import extract_reasoning
from .config import DATA_ROOT_TEMPLATE, SUBMISSION_FILENAME
from .llm_clients import CorrectorClient
from .logging_utils import get_logger
from . import prompts

LOGGER = get_logger(__name__)

#: How much of a traceback is shown to the repair model. Only the tail carries
#: the actual exception; the head is stack frames.
ERROR_TAIL_CHARS = 200

#: How much of each error is kept when several are summarised at once.
ERROR_EXCERPT_HEAD = 200
ERROR_EXCERPT_TAIL = 200
ERROR_EXCERPT_THRESHOLD = 400


def data_root(competition_name: str) -> str:
    """Relative data directory handed to every generated program."""
    return DATA_ROOT_TEMPLATE.format(competition=competition_name)


def build_timeout_optimization_prompt(
    code: str, error_message: str, competition_name: str
) -> str:
    """Ask the repair model to make a program fast enough to finish in budget."""
    return prompts.active().TIMEOUT_OPTIMIZATION_TEMPLATE.format(
        error_message=error_message,
        code=code,
        data_root=data_root(competition_name),
        submission=SUBMISSION_FILENAME,
    )


def build_generic_correction_prompt(code: str, error_message: str, competition_name: str) -> str:
    """Ask the repair model to fix an ordinary runtime failure."""
    return prompts.active().GENERIC_CORRECTION_TEMPLATE.format(
        error_tail=error_message[-ERROR_TAIL_CHARS:],
        code=code,
        data_root=data_root(competition_name),
        submission=SUBMISSION_FILENAME,
    )


def summarise_errors(
    corrector: CorrectorClient, error_messages: List[str]
) -> Optional[Tuple[str, float]]:
    """Condense an error history into a short list of failure patterns.

    Returns ``None`` when there is not enough history to generalise from; a
    single error is better handled by the direct repair path.
    """
    if not error_messages or len(error_messages) <= 1:
        return None

    errors_text = ""
    for index, error in enumerate(error_messages, start=1):
        if len(error) > ERROR_EXCERPT_THRESHOLD:
            excerpt = error[:ERROR_EXCERPT_HEAD] + "..." + error[-ERROR_EXCERPT_TAIL:]
        else:
            excerpt = error
        errors_text += f"\n\n---ERROR {index}---\n{excerpt}"

    analysis, tokens = corrector.complete(
        prompt=prompts.active().ERROR_ANALYSIS_TEMPLATE.format(errors_text=errors_text),
        system_message=prompts.error_analysis_system_prompt(),
        mode="analysis",
    )
    return analysis, tokens


def build_error_informed_task_prompt(
    competition_name: str, original_prompt: str, error_analysis: str
) -> str:
    """Restate the task with an explicit list of mistakes to avoid."""
    return prompts.active().IMPROVED_TASK_TEMPLATE.format(
        competition_name=competition_name,
        original_prompt=original_prompt,
        data_root=data_root(competition_name),
        submission=SUBMISSION_FILENAME,
        error_analysis=error_analysis,
    )


def build_regeneration_prompt(
    competition_name: str,
    original_question: str,
    regeneration_reason: Optional[str] = None,
) -> str:
    """Ask the policy model for a fresh solution after a failed attempt."""
    reason_text, focus_text = prompts.regeneration_reason_text(regeneration_reason)
    return prompts.active().REGENERATION_TEMPLATE.format(
        original_question=original_question,
        reason_text=reason_text,
        focus_text=focus_text,
        data_root=data_root(competition_name),
        submission=SUBMISSION_FILENAME,
    )


def build_optimization_prompt(
    competition_name: str,
    original_prompt: str,
    previous_response: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Ask the policy model to improve a solution that already ran successfully.

    The previous reasoning trace is carried over so that the model can build on
    its own analysis instead of restarting from the task description.
    """
    thinking = extract_reasoning(previous_response)
    if not thinking:
        thinking = (
            "No reasoning trace could be recovered; work from the task "
            "requirements instead."
        )

    if metrics and isinstance(metrics, dict):
        metrics_info = prompts.active().METRICS_KNOWN_TEMPLATE.format(
            public_score=metrics.get("public_score", "unknown"),
            private_score=metrics.get("private_score", "unknown"),
        )
    else:
        metrics_info = prompts.active().METRICS_UNKNOWN_TEMPLATE

    return prompts.active().OPTIMIZATION_TEMPLATE.format(
        original_prompt=original_prompt,
        metrics_info=metrics_info,
        thinking=thinking,
        data_root=data_root(competition_name),
        submission=SUBMISSION_FILENAME,
    )
