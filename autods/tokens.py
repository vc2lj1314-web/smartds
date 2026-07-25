"""Token accounting.

Two estimators are used, matching the two APIs the harness talks to:

* :func:`estimate_tokens_by_chars` -- a provider-agnostic character heuristic,
  used for the auxiliary repair model, whose tokeniser is not available locally.
* :func:`count_chat_tokens` -- a ``tiktoken`` count, used for the policy model.

Both are estimates. They are reported in the run record so that the token cost
of a run can be compared across configurations; they are not billing figures.
"""

from __future__ import annotations

from typing import Any, Dict, List

import tiktoken

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

#: Average number of characters per token assumed by the character heuristic.
CHARS_PER_TOKEN = 4.0

#: Fixed per-message and per-reply overheads of the OpenAI chat format.
TOKENS_PER_MESSAGE = 4
TOKENS_PER_NAME = 1
TOKENS_PER_REPLY = 3


def estimate_tokens_by_chars(messages: List[Dict[str, Any]]) -> float:
    """Estimate the token count of a chat payload from its character count."""
    total_chars = sum(
        len(str(message.get("content", "")))
        for message in messages
        if isinstance(message, dict)
    )
    return total_chars / CHARS_PER_TOKEN


def count_chat_tokens(model: str, messages: List[Dict[str, Any]]) -> float:
    """Count the tokens of a chat payload with ``tiktoken``.

    Falls back to the character heuristic when the model name is unknown to
    ``tiktoken``, which is the normal case for a locally served fine-tune.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:  # unknown model name -> no local tokeniser
        return estimate_tokens_by_chars(messages)

    num_tokens = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        num_tokens += TOKENS_PER_MESSAGE
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += TOKENS_PER_NAME
    return num_tokens + TOKENS_PER_REPLY


def estimate_completion_tokens(text: str) -> float:
    """Estimate the token count of a completion from its character count."""
    return len(text) / CHARS_PER_TOKEN
