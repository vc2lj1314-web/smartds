"""Prompt-set selection.

Two complete prompt sets ship with this repository:

``prompts_zh``
    The **experiment of record**: the exact Chinese wording used to produce the
    results reported in the paper. This is the default.

``prompts_en``
    A faithful English translation, provided so that a reader who does not read
    Chinese can follow what the agent asks of each model.

They are not interchangeable. The evaluated policy model was instruction-tuned
on Chinese reasoning traces, so switching sets changes the system under
measurement -- most visibly the rate at which the model honours the ``<answer>``
output contract, which in turn feeds the decoding-parameter controller. Treat
``--prompt-language en`` as an ablation, not as a reproduction.

Everything downstream reaches the active set through :func:`active` rather than
importing template names directly, so the selection can be made once at start-up
and still take effect everywhere.
"""

from __future__ import annotations

from types import ModuleType
from typing import Dict, Optional, Tuple

from . import prompts_en, prompts_zh
from .config import DATA_ROOT_TEMPLATE, PROMPT_LANGUAGE, SUBMISSION_FILENAME
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

#: Every template name a prompt set must define.
REQUIRED_SYMBOLS = (
    "GENERATION_SYSTEM_PROMPT",
    "OPTIMIZATION_SYSTEM_PROMPT",
    "CODE_CORRECTION_SYSTEM_PROMPT",
    "ERROR_ANALYSIS_SYSTEM_PROMPT",
    "TIMEOUT_OPTIMIZATION_TEMPLATE",
    "GENERIC_CORRECTION_TEMPLATE",
    "ERROR_ANALYSIS_TEMPLATE",
    "IMPROVED_TASK_TEMPLATE",
    "REGENERATION_TEMPLATE",
    "OPTIMIZATION_TEMPLATE",
    "METRICS_KNOWN_TEMPLATE",
    "METRICS_UNKNOWN_TEMPLATE",
    "REGENERATION_REASON_TEXT",
    "DEFAULT_REGENERATION_REASON_TEXT",
)

PROMPT_SETS: Dict[str, ModuleType] = {"zh": prompts_zh, "en": prompts_en}


def _validate(prompt_set: ModuleType) -> None:
    """Fail loudly if a prompt set is missing a template the pipeline needs."""
    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(prompt_set, name)]
    if missing:
        raise AttributeError(
            f"Prompt set {prompt_set.__name__} is missing: {', '.join(missing)}"
        )


for _prompt_set in PROMPT_SETS.values():
    _validate(_prompt_set)

if PROMPT_LANGUAGE not in PROMPT_SETS:
    raise ValueError(
        f"Unknown prompt language {PROMPT_LANGUAGE!r}; expected one of {sorted(PROMPT_SETS)}"
    )

_active_language = PROMPT_LANGUAGE
_active: ModuleType = PROMPT_SETS[PROMPT_LANGUAGE]


def active() -> ModuleType:
    """Return the prompt set currently in use."""
    return _active


def active_language() -> str:
    """Return the language code of the prompt set currently in use."""
    return _active_language


def set_prompt_language(language: str) -> ModuleType:
    """Switch the active prompt set. Call this once, before an experiment starts."""
    if language not in PROMPT_SETS:
        raise ValueError(
            f"Unknown prompt language {language!r}; expected one of {sorted(PROMPT_SETS)}"
        )

    global _active, _active_language
    _active_language = language
    _active = PROMPT_SETS[language]

    if language != "zh":
        LOGGER.warning(
            "Prompt language set to %r. The reported results were produced with "
            "the Chinese set; this run is not a reproduction of them.",
            language,
        )
    else:
        LOGGER.info("Prompt language set to 'zh' (the experiment of record)")

    return _active


# --------------------------------------------------------------------------- #
# Accessors used by the rest of the pipeline
# --------------------------------------------------------------------------- #


def generation_system_prompt() -> str:
    """System prompt for first-attempt generation and for regeneration."""
    return _active.GENERATION_SYSTEM_PROMPT


def optimization_system_prompt() -> str:
    """System prompt for improving a solution that already ran successfully."""
    return _active.OPTIMIZATION_SYSTEM_PROMPT


def error_analysis_system_prompt() -> str:
    """System prompt for the auxiliary model when summarising a failure history."""
    return _active.ERROR_ANALYSIS_SYSTEM_PROMPT


def code_correction_system_prompt(competition_name: str) -> str:
    """System prompt for the auxiliary repair model, bound to a task's data path."""
    return _active.CODE_CORRECTION_SYSTEM_PROMPT.format(
        submission=SUBMISSION_FILENAME,
        data_root=DATA_ROOT_TEMPLATE.format(competition=competition_name),
    )


def regeneration_reason_text(regeneration_reason: Optional[str]) -> Tuple[str, str]:
    """Return ``(reason_text, focus_text)`` for a regeneration trigger."""
    return _active.REGENERATION_REASON_TEXT.get(
        regeneration_reason or "", _active.DEFAULT_REGENERATION_REASON_TEXT
    )
