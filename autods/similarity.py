"""Semantic deadlock detection (Algorithm 1, line 32).

If the current error log is semantically close to one already seen in this
trajectory, the repair operator is rewriting the program without addressing the
defect, and further repairs only burn tokens. Algorithm 1 evaluates

    ErrorSim(Log(E_t,k), H) > tau_sim

before every LocalRepair call, comparing against the whole history rather than
against a single earlier error. The comparison uses a small multilingual
sentence encoder on the last 100 characters of each message, which is where the
concrete exception text lives; the head of a traceback is mostly frame noise.
"""

from __future__ import annotations

from typing import List, Sequence

from sklearn.metrics.pairwise import cosine_similarity

from .config import SIMILARITY_MODEL_NAME
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

#: Number of trailing characters of each message that are compared.
ERROR_TAIL_CHARS = 100

_encoder = None  # lazily loaded; the model is ~120 MB


def _get_encoder():
    """Load the sentence encoder on first use."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        LOGGER.info("Loading sentence encoder %s", SIMILARITY_MODEL_NAME)
        _encoder = SentenceTransformer(SIMILARITY_MODEL_NAME)
    return _encoder


def error_sim(current_error: str, history: Sequence[str]) -> float:
    """``ErrorSim(Log(E), H)`` -- max similarity of ``current_error`` to ``history``.

    Returns 0.0 for an empty history, i.e. "no evidence of a deadlock yet".
    """
    if not history or not current_error:
        return 0.0

    tails: List[str] = [current_error[-ERROR_TAIL_CHARS:]]
    tails += [message[-ERROR_TAIL_CHARS:] for message in history]

    embeddings = _get_encoder().encode(tails)
    similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]
    return float(max(similarities))


def max_pairwise_similarity(error_messages: Sequence[str]) -> float:
    """Max similarity between the newest error and every earlier one.

    Convenience wrapper used when classifying a finished trajectory.
    """
    if len(error_messages) < 2:
        return 0.0
    return error_sim(error_messages[-1], error_messages[:-1])
