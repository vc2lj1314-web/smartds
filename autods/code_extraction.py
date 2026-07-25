"""Recovering an executable program from a free-form model response.

The policy model is asked to wrap its program in ``<answer>`` tags, but a
fraction of responses do not comply: the tags are missing, the program is split
across several fenced blocks, or the program is buried under a long prose
explanation. Extraction therefore proceeds through a cascade of strategies,
from the most reliable to the most speculative:

1. fenced code blocks inside ``<answer>`` ... ``</answer>``
2. an indentation/keyword scan inside ``<answer>``
3. fenced blocks that follow a "complete solution"-style heading
4. every fenced block in the response, de-duplicated and concatenated
5. an indentation/keyword scan over the whole response

Whether extraction succeeded via the ``<answer>`` contract is reported back to
the caller, because the parameter controller rewards format compliance.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .config import (
    CJK_PROSE_MARKERS,
    CJK_SOLUTION_HEADING_PATTERNS,
    INCLUDE_CJK_EXTRACTION_MARKERS,
)
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)\s*```")
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
OPEN_THINK_RE = re.compile(r"<think>(.*?)(</think>|$)", re.DOTALL)

MIN_CODE_LENGTH = 50
MAX_EXPLANATION_RATIO = 0.4
MIN_LINES_BEFORE_STOP = 10
MIN_LINES_TO_KEEP = 5

#: A response longer than this is treated as unstructured rambling.
VERBOSE_HARD_LINE_LIMIT = 600
VERBOSE_HARD_CHAR_LIMIT = 30000
VERBOSE_SOFT_LINE_LIMIT = 400
VERBOSE_SOFT_CHAR_LIMIT = 20000
MAX_THINK_RATIO = 0.7

#: Phrases that mark a line as prose rather than code. The CJK counterpart lives
#: in :mod:`autods.config` behind ``INCLUDE_CJK_EXTRACTION_MARKERS``, because the
#: policy model may answer in Chinese even when prompted in English and dropping
#: those markers silently degrades extraction on such responses.
ENGLISH_PROSE_MARKERS = (
    "needs verification", "for example", "assume", "therefore", "and so on",
    "this is a key point", "the original training data", "the test data",
    "when merging", "then", "according to", "next", "parse", "extract",
    "because", "needs to be grouped", "key point", "match", "convert", "merge",
    "explanation", "note that", "in summary", "the approach", "improvement",
    "optimisation", "performance", "metric",
)

PROSE_MARKERS = ENGLISH_PROSE_MARKERS + (
    CJK_PROSE_MARKERS if INCLUDE_CJK_EXTRACTION_MARKERS else ()
)

#: Headings a verbose response tends to put immediately before the final program.
ENGLISH_SOLUTION_HEADING_PATTERNS = (
    r"###?\s*complete (?:improved )?solution\s*$",
    r"###?\s*complete implementation\s*$",
    r"##?\s*full implementation\s*$",
    r"###?\s*final implementation\s*$",
    r"###?\s*implementation\s*$",
    r"###?\s*complete code\s*$",
    r"###?\s*final code\s*$",
)

SOLUTION_HEADING_PATTERNS = ENGLISH_SOLUTION_HEADING_PATTERNS + (
    CJK_SOLUTION_HEADING_PATTERNS if INCLUDE_CJK_EXTRACTION_MARKERS else ()
)


CODE_LINE_STARTERS = (
    "import ", "from ", "def ", "class ", "if ", "for ", "while ",
    "try:", "except", "elif ", "else:", "return ", "print(",
)

CODE_CALL_MARKERS = ("plt.", "pd.", "np.", "df.", "model.", "keras.", "tensorflow.")

DATA_API_MARKERS = (".fit(", ".predict(", ".transform(", ".read_csv(")


def _is_prose(line: str) -> bool:
    """True when the line reads as natural-language explanation."""
    lowered = line.lower()
    return any(marker.lower() in lowered for marker in PROSE_MARKERS)


def looks_like_complete_program(code: str) -> bool:
    """Heuristic guard against extracting a prose fragment or a code snippet.

    A candidate passes when it is long enough, is not mostly prose, and contains
    both an import and an assignment -- the minimum for a self-contained script.
    """
    if not code or len(code.strip()) < MIN_CODE_LENGTH:
        return False

    lines = [line.strip() for line in code.split("\n") if line.strip()]

    explanation_lines = 0
    code_lines = 0
    for line in lines:
        if _is_prose(line):
            explanation_lines += 1
        elif (
            any(line.startswith(starter) for starter in CODE_LINE_STARTERS)
            or "=" in line
            or line.endswith(":")
            or "print(" in line
            or any(marker in line for marker in DATA_API_MARKERS)
        ):
            code_lines += 1

    if explanation_lines > 0 and code_lines > 0:
        ratio = explanation_lines / (explanation_lines + code_lines)
        if ratio > MAX_EXPLANATION_RATIO:
            LOGGER.debug("Rejected candidate: explanation ratio %.2f", ratio)
            return False

    has_import = any("import" in line for line in lines[:15])
    has_assignment = any("=" in line for line in lines)
    if not (has_import and has_assignment):
        LOGGER.debug(
            "Rejected candidate: import=%s assignment=%s", has_import, has_assignment
        )
        return False

    return True


def _scan_code_lines(text: str, stop_on_heading: bool = False) -> List[str]:
    """Collect consecutive code-looking lines from ``text``.

    The scan starts at the first import / def / class line and then keeps lines
    that are blank, indented, or otherwise code-shaped. It stops at the first
    clear prose line, unless too few lines have been collected so far, in which
    case the run is treated as a false start and reset.
    """
    collected: List[str] = []
    in_code_section = False

    for line in text.split("\n"):
        stripped = line.strip()

        if stop_on_heading and stripped.startswith("#") and not stripped.startswith("# "):
            break

        if stripped.startswith(("import ", "from ", "def ", "class ")) or stripped.startswith(
            "if __name__"
        ):
            in_code_section = True
            collected.append(line)
            continue

        if not in_code_section:
            continue

        if (
            stripped == ""
            or stripped.startswith("#")
            or line.startswith(("    ", "\t"))
            or "=" in stripped
            or stripped.endswith(":")
            or any(stripped.startswith(starter) for starter in CODE_LINE_STARTERS)
            or any(marker in stripped for marker in CODE_CALL_MARKERS)
        ):
            collected.append(line)
        elif stripped and not _is_prose(stripped):
            # Ambiguous line, but nothing marks it as prose: keep it.
            collected.append(line)
        else:
            non_empty = len([item for item in collected if item.strip()])
            if non_empty >= MIN_LINES_BEFORE_STOP:
                break
            if non_empty < MIN_LINES_TO_KEEP:
                collected = []
                in_code_section = False

    return collected


def _extract_after_heading(text: str, heading_patterns) -> Optional[str]:
    """Return the program that follows one of ``heading_patterns``, if any."""
    for pattern in heading_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            continue

        remainder = text[match.end():]

        fenced = CODE_FENCE_RE.search(remainder)
        if fenced:
            candidate = fenced.group(1).strip()
            if looks_like_complete_program(candidate):
                return candidate

        scanned = _scan_code_lines(remainder, stop_on_heading=True)
        if scanned:
            candidate = "\n".join(scanned)
            if looks_like_complete_program(candidate):
                return candidate

    return None


def _merge_all_fenced_blocks(text: str) -> Optional[str]:
    """Concatenate every distinct fenced block that looks like real code."""
    blocks = CODE_FENCE_RE.findall(text)
    if not blocks:
        return None

    valid_blocks: List[str] = []
    seen_signatures = set()

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 3:
            continue

        # De-duplicate on the first three significant lines: models often repeat
        # the same program once as a draft and once as the final answer.
        signature = "\n".join(lines[:3])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        if looks_like_complete_program(block):
            valid_blocks.append(block)
        elif len(lines) >= 5 and any("import" in line or "=" in line for line in lines[:5]):
            # Short helper block: keep it, it may complete the main one.
            valid_blocks.append(block)

    return "\n\n".join(valid_blocks) if valid_blocks else None


def is_response_too_verbose(text: str) -> bool:
    """True when a response is so long and unstructured that extraction is unsafe.

    Beyond a hard length limit the response is rejected outright. Between the
    soft and hard limits it is rejected only if it also lacks structure, i.e.
    almost no fenced blocks and few headings, or if the reasoning trace makes up
    most of the response and the program was never written out.
    """
    lines = text.split("\n")
    total_lines = len(lines)
    total_chars = len(text)

    if total_lines > VERBOSE_HARD_LINE_LIMIT or total_chars > VERBOSE_HARD_CHAR_LIMIT:
        return True

    if total_lines > VERBOSE_SOFT_LINE_LIMIT or total_chars > VERBOSE_SOFT_CHAR_LIMIT:
        think_match = OPEN_THINK_RE.search(text)  # tolerate an unclosed <think>
        if think_match and total_chars:
            think_ratio = len(think_match.group(1)) / total_chars
            if think_ratio > MAX_THINK_RATIO:
                return True

        fence_count = len(re.findall(r"```", text))
        heading_count = len(re.findall(r"^#{1,4}\s", text, re.MULTILINE))
        if fence_count < 1 and heading_count < 4:
            return True

    return False


def extract_program(response: str) -> Tuple[Optional[str], bool]:
    """Extract an executable program from a model response.

    Returns:
        ``(program, from_answer_tag)``. ``program`` is ``None`` when no strategy
        produced something that looks complete. ``from_answer_tag`` records
        whether the model honoured the ``<answer>`` output contract, which the
        parameter controller uses as a format-compliance signal.
    """
    answer_match = ANSWER_TAG_RE.search(response)
    if answer_match:
        answer_text = answer_match.group(1).strip()

        blocks = CODE_FENCE_RE.findall(answer_text)
        if blocks:
            combined = "\n\n".join(blocks)
            if looks_like_complete_program(combined):
                return combined, True

        scanned = _scan_code_lines(answer_text)
        if scanned:
            candidate = "\n".join(scanned)
            if looks_like_complete_program(candidate):
                return candidate, True

    if is_response_too_verbose(response):
        LOGGER.warning("Response is long and unstructured; no program extracted")
        return None, False

    after_heading = _extract_after_heading(response, SOLUTION_HEADING_PATTERNS)
    if after_heading:
        return after_heading, False

    merged = _merge_all_fenced_blocks(response)
    if merged:
        return merged, False

    scanned = _scan_code_lines(response)
    if scanned:
        candidate = "\n".join(scanned)
        if looks_like_complete_program(candidate):
            return candidate, False

    return None, False


def extract_fenced_code(text: str) -> str:
    """Permissive extraction used on the auxiliary repair model's output.

    The repair model is asked for code only, so its response is either fenced or
    already bare. Every fenced block is concatenated; if there is no fence, the
    text is filtered down to code-looking lines and returned as-is on failure,
    because the caller falls back to the raw response anyway.
    """
    blocks = CODE_FENCE_RE.findall(text)
    if blocks:
        if len(blocks) > 1:
            LOGGER.info("Merging %d fenced blocks from the repair response", len(blocks))
        return "\n\n".join(blocks)

    cleaned: List[str] = []
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            continue
        if line.strip().startswith("#") or "import" in line or "def " in line or "=" in line:
            in_code = True
        if in_code:
            cleaned.append(line)

    return "\n".join(cleaned) if cleaned else text


def extract_reasoning(response: str) -> str:
    """Return the contents of the ``<think>`` block, or an empty string."""
    match = THINK_TAG_RE.search(response)
    return match.group(1).strip() if match else ""
