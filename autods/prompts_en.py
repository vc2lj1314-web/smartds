"""Prompt templates, English — a translation provided for readability.

This set is a faithful translation of :mod:`autods.prompts_zh`, which is what
actually produced the results reported in the paper. It exists so that a reader
who does not read Chinese can follow exactly what the agent asks of each model.

Selecting it changes the experiment. The policy model was instruction-tuned on
Chinese reasoning traces, so running with ``--prompt-language en`` measures a
different system: expect a different rate of ``<answer>``-contract compliance
and, through it, a different trajectory of the decoding-parameter controller.
Use it for inspection, or as a deliberate ablation on prompt language — not to
reproduce the reported numbers.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

#: Used for the first generation of every attempt. It fixes the think/answer
#: output contract that the code extractor depends on.
GENERATION_SYSTEM_PROMPT = (
    "You are a rigorous AI assistant. You must answer strictly in the following "
    "format:\n"
    "1. Put your reasoning between <think> and </think>, analysing the problem "
    "step by step in plain, conversational language.\n"
    "2. Put the final answer between <answer> and </answer>.\n\n"
    "Format example:\n"
    "<think>\n"
    "Okay, I need to solve this problem. First let me confirm...\n"
    "Wait, what should I do if...? Next I should...\n"
    "For instance..., and finally I combine all the conditions.\n"
    "</think>\n\n"
    "<answer>\nThe final answer goes here\n</answer>\n\n"
    "Please note:\n"
    "- Open with a discourse marker such as 'Okay' or 'Right'.\n"
    "- Use 'First', 'Next' and 'Then' to connect the steps.\n"
    "- Use 'Wait' to introduce a caveat or an open question.\n"
    "Now answer the user's question:"
)

#: Used when the previous attempt executed successfully and the goal is to
#: improve the score rather than to fix a failure.
OPTIMIZATION_SYSTEM_PROMPT = (
    "You are an experienced machine learning engineer, fluent in data science "
    "competition solutions. Improve the existing solution using the information "
    "below:\n"
    "1. Analyse the metrics and the errors of the previous solution.\n"
    "2. Exploit the insights recorded in the original reasoning trace.\n"
    "3. Consider changes to the model architecture, the feature engineering and "
    "the hyper-parameters.\n"
    "4. Produce one complete, directly executable Python program.\n\n"
    "Answer format:\n"
    "<think>\nA detailed analysis of the weaknesses of the current solution and "
    "of the possible improvements\n</think>\n\n"
    "<answer>\nThe complete improved program\n</answer>"
)

#: Terse system prompt for the auxiliary repair model. It asks for code only,
#: because the repair path does not need a reasoning trace.
CODE_CORRECTION_SYSTEM_PROMPT = (
    "You are a Python code repair expert. Requirements:\n"
    "1. Analyse the error and fix the program.\n"
    "2. Return only the complete fixed program, with no explanation.\n"
    "3. The program must be complete and executable.\n"
    "4. The program must write {submission}.\n"
    "5. Do not wrap the submission-writing logic in try/except.\n"
    "6. IMPORTANT: read the data strictly from '{data_root}'. Never invent or "
    "simulate data.\n\n"
    "Format: return the fixed Python program directly."
)

#: System prompt for the auxiliary model when it is asked to summarise a series
#: of failures instead of repairing a program.
ERROR_ANALYSIS_SYSTEM_PROMPT = (
    "You are an efficiency expert who analyses code errors concisely and "
    "precisely and reports the findings as short bullet points."
)


# --------------------------------------------------------------------------- #
# User prompt templates
# --------------------------------------------------------------------------- #

TIMEOUT_OPTIMIZATION_TEMPLATE = """The program below timed out and must be made more efficient.

Error message:
{error_message}

Original program:
```python
{code}
```

Optimise the program so that it finishes in time. Possible strategies:
1. Simplify the network architecture (fewer layers, fewer units).
2. Reduce the number of epochs or the batch size.
3. Fall back to a simpler machine learning algorithm.
4. Streamline the preprocessing pipeline.
5. Reduce the number of features or the number of samples.
6. Use a more efficient implementation of the same algorithm.

Requirements:
- Data path: {data_root}
- The program must write {submission}.
- The program must finish within a reasonable wall-clock budget.
- Do not wrap the submission-writing logic in try/except.

Return the complete optimised Python program directly."""


GENERIC_CORRECTION_TEMPLATE = """Error: {error_tail}

Fix the program below and return the complete executable version:
```python
{code}
```

Requirements: the data is in {data_root}, and the program must write {submission}."""


ERROR_ANALYSIS_TEMPLATE = """Analyse the Python execution errors below and list the key problems and \
suggested fixes as concise bullet points:
{errors_text}

Answer briefly:
1. Shared failure patterns (at most 3 points, at most 30 words each)
2. Main problem categories (at most 2, at most 15 words each)
3. Key suggested fixes (at most 4 points, at most 25 words each)

Format requirements:
- No superfluous explanation
- A short bullet list
- At most 300 words in total
- Plain text, no markup"""


IMPROVED_TASK_TEMPLATE = """# Data science task: {competition_name}

## Task description and requirements
{original_prompt}

## Implementation notes
Provide a complete solution that achieves the objective above. Your program must:

1. Correctly understand and complete the core objective of the task.
2. Read the data from '{data_root}'.
3. Write a {submission} file in the format the task requires.
4. Follow data science best practice for preprocessing, feature engineering and \
model selection.

## Mistakes to avoid
The following problems were observed in earlier attempts. Avoid them when you \
design the solution:

{error_analysis}

## Code requirements
- Provide a complete, executable Python program, including every import.
- Add clear comments explaining the key steps and the reasoning behind them.
- Prefer stable, well-understood methods over high-risk experimental ones.
- Make sure the program runs efficiently and neither times out nor exhausts memory.
- IMPORTANT: never wrap the submission-writing logic in try/except; {submission} \
must be produced reliably.

Remember: the primary goal is to complete the data science task and produce \
valid predictions, with a program that executes reliably."""


REGENERATION_TEMPLATE = """Generate a complete solution for the data science task below from scratch.

## Original task description
{original_question}

## Reason for regeneration
{reason_text}. The solution needs to be redesigned.

## Data location
{data_root}

## Requirements for the solution
{focus_text}, and make sure it runs correctly and writes {submission}.

### Specific requirements:
1. **Completeness**: a complete, executable Python program including every import.
2. **Robustness**: handle edge cases and exceptions.
3. **Output format**: write {submission} in the required format.
4. **Comments**: explain the key steps and the reasoning behind them.
5. **Library choice**: prefer the most basic, stable libraries and methods; avoid \
over-complicated implementations.
6. **Runtime**: make sure the program finishes within a reasonable budget.
7. **IMPORTANT**: never wrap the submission-writing logic in try/except; \
{submission} must be produced reliably.

### Implementation strategy:
- Use classical, well-validated machine learning methods.
- Keep the preprocessing pipeline simple and explicit.
- Give every step a clear purpose.
- Avoid over-engineering; focus on the core problem.

Think the task through carefully, then provide the complete executable Python program."""


OPTIMIZATION_TEMPLATE = """Improve the solution to the data science task below.

### Task description:
{original_prompt}

### Current status:
{metrics_info}

### Original reasoning trace:
{thinking}

Analyse the current solution and propose improvements. Focus on:
1. Whether the feature engineering is sufficient
2. Whether the model choice is appropriate
3. Whether the hyper-parameters need tuning
4. Whether there is a potential data leakage problem
5. Whether an ensemble of several models would help
6. Whether the cross-validation strategy is sound

Produce a complete and more effective Python program. It must differ \
substantially from the current one and must improve performance.
The program must be complete and executable, must read its data from \
"{data_root}", and must write the {submission} file the task requires.

Note in particular: never wrap the submission-writing logic in try/except; \
{submission} must be produced reliably."""


METRICS_KNOWN_TEMPLATE = """Leaderboard performance of the current model:
- Public test score: {public_score}
- Private test score: {private_score}

Analyse these scores and consider how to improve the ranking.
"""


METRICS_UNKNOWN_TEMPLATE = """No leaderboard score is available for the previous model, so focus on the \
following optimisation strategies:
- Improve the feature engineering
- Try a more capable model architecture
- Tune the hyper-parameters
- Look for headroom in the preprocessing pipeline

"""


#: Human-readable explanation of why a solution is being regenerated, indexed by
#: the machine-readable reason recorded in the attempt record.
REGENERATION_REASON_TEXT = {
    "correction_failed": (
        "the previous program kept hitting the same problem during execution",
        "Generate a simpler and more robust solution",
    ),
    "extract_failed": (
        "no program could be extracted from the previous response, so more "
        "diverse sampling is needed",
        "Generate a more inventive solution with a clearly delimited program",
    ),
}

DEFAULT_REGENERATION_REASON_TEXT = (
    "the previous program ran into a problem during execution",
    "Generate a more stable and reliable solution",
)
