"""The self-correction loop, expressed as a LangGraph state machine.

Two nodes and one router:

* ``correct_code`` runs the current program and, on failure, asks the auxiliary
  model for a repair.
* ``restore_epochs`` is entered once a deep-learning program has survived its
  epochs=1 smoke test; it restores the original epoch count and runs the full
  training job under a timeout extrapolated from the smoke-test runtime.
* :func:`route_after_step` decides whether to iterate, move to full training or
  stop.

The loop stops on any of: success, a repeated timeout, a stalled repair loop
(the error stopped changing), or the correction budget being exhausted.
"""

from __future__ import annotations

import re
import traceback
from typing import Literal, Optional

from .code_analysis import (
    detect_dl_code_and_epochs,
    is_timeout_error,
    restore_epoch_count,
)
from .code_extraction import extract_fenced_code
from .config import (
    FULL_TRAINING_TIMEOUT_MARGIN,
    MAX_CONSECUTIVE_TIMEOUTS,
    N_REPAIR,
    SUBMISSION_FILENAME,
    TAU_SIM,
    ExecutionConfig,
)
from .execution import run_program
from .llm_clients import CorrectorClient
from .logging_utils import get_logger
from .prompt_builders import (
    build_generic_correction_prompt,
    build_timeout_optimization_prompt,
    data_root,
)
from .prompts import code_correction_system_prompt
from .similarity import error_sim
from .state import CorrectionState
from .telemetry import RunRecorder

LOGGER = get_logger(__name__)

MIN_USABLE_CODE_LENGTH = 10


def enforce_data_path(code: str, competition_name: str) -> str:
    """Rewrite every input path in ``code`` to the canonical task data directory.

    The generated program is free to write ``input/``, ``./input/`` or
    ``../input/``; all of them are normalised to ``../input/<task>/``.

    The rewrite is a single regular-expression pass with a negative lookahead on
    the task name. A sequence of independent ``str.replace`` calls, as in the
    original implementation, is not idempotent: normalising ``../input/`` first
    and then replacing the bare substring ``input/`` re-rewrites the result and
    produces a doubled path such as ``../../input/<task>/<task>/``.
    """
    canonical = data_root(competition_name)
    pattern = re.compile(r"(?:\.\./|\./)?input/(?!" + re.escape(competition_name) + r"/)")
    return pattern.sub(canonical, code)


def _select_timeout(code: str, state: CorrectionState) -> int:
    """Pick the wall-clock budget for the next execution of ``code``.

    A deep-learning program that is still in its epochs=1 smoke test gets the
    short budget; everything else gets the ordinary one.
    """
    is_dl_code, epochs_value, _ = detect_dl_code_and_epochs(code)
    if is_dl_code and epochs_value == 1:
        return state.get("dl_testing_timeout")
    return state.get("normal_timeout")


class CorrectionLoop:
    """Builds a compiled correction graph bound to one corrector and config.

    LangGraph nodes receive only the state, so the dependencies the nodes need
    (the auxiliary model client, the execution budget, the telemetry sink) are
    captured here instead of being module-level globals.
    """

    def __init__(
        self,
        corrector: CorrectorClient,
        execution: ExecutionConfig,
        recorder: Optional[RunRecorder] = None,
    ) -> None:
        self.corrector = corrector
        self.execution = execution
        self.recorder = recorder

    # -- helpers ------------------------------------------------------------ #

    def _repair(self, prompt: str, competition_name: str) -> str:
        """Send one repair request and return the extracted program."""
        response, tokens = self.corrector.complete(
            prompt=prompt,
            system_message=code_correction_system_prompt(competition_name),
            mode="correction",
        )
        if self.recorder is not None:
            self.recorder.add_correction_tokens(tokens)

        repaired = extract_fenced_code(response)
        if not repaired or len(repaired.strip()) < MIN_USABLE_CODE_LENGTH:
            LOGGER.warning("Repair response contained no usable program; using it verbatim")
            repaired = response
        return repaired

    def _accept_repair(self, state: CorrectionState, repaired: str) -> None:
        """Normalise a repaired program and install it as the current one."""
        repaired = enforce_data_path(repaired, state["competition_name"])

        if state["is_dl_code"] and state["testing_phase"]:
            _, _, smoke_test_code = detect_dl_code_and_epochs(repaired)
            if smoke_test_code:
                repaired = smoke_test_code
                LOGGER.info("Kept epochs=1 so the repaired pipeline is smoke-tested first")

        state["corrections"].append(repaired)
        state["current_code"] = repaired
        state["correction_count"] += 1

    # -- nodes -------------------------------------------------------------- #

    def correct_code(self, state: CorrectionState) -> CorrectionState:
        """Run the current program; on failure, request one repair."""
        state.setdefault("timeout_count", 0)
        competition_name = state.get("competition_name", "unknown-competition")
        current_code = state["current_code"]

        result = run_program(
            current_code,
            timeout_seconds=_select_timeout(current_code, state),
            cpu_limit=self.execution.cpu_limit,
        )

        if state["is_dl_code"] and state["testing_phase"] and result.execution_time is not None:
            state["test_execution_time"] = result.execution_time
            LOGGER.info("Smoke test finished in %.1f s", result.execution_time)

        if result.success:
            state["final_success"] = True
            if state["is_dl_code"] and state["testing_phase"]:
                state["test_success"] = True
                LOGGER.info("Smoke test passed; will rerun with the original epoch count")
            else:
                state["status"] = "completed"
            return state

        error_message = result.output
        state["error_messages"].append(error_message)
        state["final_success"] = False

        if result.timed_out or is_timeout_error(error_message):
            return self._handle_timeout(state, current_code, error_message, competition_name)

        try:
            prompt = build_generic_correction_prompt(current_code, error_message, competition_name)
            self._accept_repair(state, self._repair(prompt, competition_name))
            LOGGER.info(
                "Repair %d applied (timeouts so far: %d)",
                state["correction_count"],
                state["timeout_count"],
            )
        except Exception as exc:
            LOGGER.error("Repair step failed: %s", exc)
            LOGGER.debug(traceback.format_exc())
            state["error_messages"].append(f"Repair step failed: {exc}")
            state["correction_count"] += 1

        return state

    def _handle_timeout(
        self,
        state: CorrectionState,
        current_code: str,
        error_message: str,
        competition_name: str,
    ) -> CorrectionState:
        """Escalation path for timeouts.

        The first timeout is treated as an efficiency defect and sent to the
        auxiliary model with a prompt that asks for a cheaper solution. A second
        timeout means the whole approach is too expensive; the loop stops so the
        episode can spend its budget on a freshly generated solution instead.
        """
        state["timeout_count"] += 1
        LOGGER.warning("Timeout detected (count now %d)", state["timeout_count"])

        if state["timeout_count"] >= MAX_CONSECUTIVE_TIMEOUTS:
            LOGGER.warning("Timeout budget exhausted; abandoning this program")
            state["status"] = "timeout_error"
            state["correction_count"] = state.get("n_repair", N_REPAIR)  # force the router to stop
            return state

        try:
            prompt = build_timeout_optimization_prompt(
                current_code, error_message, competition_name
            )
            self._accept_repair(state, self._repair(prompt, competition_name))
            LOGGER.info("Efficiency rewrite applied after timeout %d", state["timeout_count"])
        except Exception as exc:
            LOGGER.error("Efficiency rewrite failed: %s", exc)
            state["status"] = "timeout_error"
            state["correction_count"] = state.get("n_repair", N_REPAIR)
        return state

    def restore_epochs(self, state: CorrectionState) -> CorrectionState:
        """Rerun a smoke-tested deep-learning program with its original epoch count.

        The timeout for the full run is extrapolated from the measured
        single-epoch runtime rather than fixed, so a cheap model is not given
        hours it does not need and an expensive one is not cut off at an
        arbitrary boundary.
        """
        if state.get("status") == "timeout_error" or state.get("timeout_count", 0) >= MAX_CONSECUTIVE_TIMEOUTS:
            LOGGER.warning("Timed out earlier; skipping the full training run")
            state["status"] = "timeout_error"
            return state

        if not (
            state["is_dl_code"]
            and state["test_success"]
            and state["testing_phase"]
            and state["original_epochs"]
        ):
            return state

        original_epochs = state["original_epochs"]
        LOGGER.info("Smoke test passed; restoring epochs=%d", original_epochs)

        current_code = restore_epoch_count(state["current_code"], original_epochs)
        current_code = (
            f"# Validated with a single epoch; running the full "
            f"{original_epochs}-epoch training below.\n" + current_code
        )
        state["current_code"] = current_code
        state["testing_phase"] = False

        smoke_time = state.get("test_execution_time")
        if smoke_time:
            timeout_seconds = int(smoke_time * FULL_TRAINING_TIMEOUT_MARGIN * original_epochs)
            LOGGER.info(
                "Full-training timeout extrapolated from the %.1f s smoke test: %d s",
                smoke_time,
                timeout_seconds,
            )
        else:
            timeout_seconds = state.get("normal_timeout")
            LOGGER.info("No smoke-test timing available; using the default timeout")

        result = run_program(
            current_code,
            timeout_seconds=timeout_seconds,
            cpu_limit=self.execution.cpu_limit,
        )

        if result.success:
            state["status"] = "completed"
            state["final_success"] = True
            LOGGER.info("Full training completed successfully")
            return state

        state["error_messages"].append(result.output)
        state["final_success"] = False

        if result.timed_out or is_timeout_error(result.output):
            state["timeout_count"] = state.get("timeout_count", 0) + 1
            if state["timeout_count"] >= MAX_CONSECUTIVE_TIMEOUTS:
                state["status"] = "timeout_error"
            else:
                state["status"] = "needs_correction"
        else:
            state["status"] = "needs_correction"

        LOGGER.warning("Full training failed: %s", result.output[:200])
        return state

    def build(self):
        """Compile the graph."""
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(CorrectionState)
        workflow.add_node("correct_code", self.correct_code)
        workflow.add_node("restore_epochs", self.restore_epochs)

        transitions = {
            "continue": "correct_code",
            "restore_epochs": "restore_epochs",
            "end": END,
        }
        workflow.add_conditional_edges("correct_code", route_after_step, transitions)
        workflow.add_conditional_edges("restore_epochs", route_after_step, transitions)
        workflow.set_entry_point("correct_code")

        return workflow.compile()


def route_after_step(state: CorrectionState) -> Literal["continue", "restore_epochs", "end"]:
    """Decide what the refinement loop does next (Algorithm 1, inner loop).

    The three exits correspond to the three ``break``/``return`` points of the
    inner loop: a valid execution, a semantic deadlock, and exhaustion of the
    repair budget. Stop conditions are checked before continuation conditions,
    so a terminal state can never be overridden by a later branch.

    The deadlock test runs before *every* repair, not once at a fixed round, and
    compares the newest error against the whole history rather than against a
    single earlier error. Checking once at a fixed round lets a trajectory that
    stalls on its first or last repair burn the entire budget undetected.
    """
    timeout_count = state.get("timeout_count", 0)
    status = state.get("status", "")
    correction_count = state.get("correction_count", 0)
    n_repair = state.get("n_repair", N_REPAIR)

    if status == "timeout_error" or timeout_count >= MAX_CONSECUTIVE_TIMEOUTS:
        LOGGER.info("Stopping: timeout budget exhausted")
        state["termination"] = "timeout"
        return "end"

    if state["is_dl_code"] and state["test_success"] and state["testing_phase"]:
        return "restore_epochs"

    if status == "completed" or state.get("final_success", False):
        LOGGER.info("Stopping: a valid submission was produced")
        state["termination"] = "success"
        return "end"

    errors = state["error_messages"]
    if len(errors) >= 2:
        similarity = error_sim(errors[-1], errors[:-1])
        state["last_similarity"] = similarity
        LOGGER.info("ErrorSim(newest, history) = %.4f (tau_sim = %.2f)", similarity, TAU_SIM)
        if similarity > TAU_SIM:
            LOGGER.info("Stopping: semantic deadlock, the error stopped changing")
            state["deadlock"] = True
            state["termination"] = "deadlock"
            return "end"

    if correction_count >= n_repair:
        LOGGER.info("Stopping: repair budget exhausted (N_repair = %d)", n_repair)
        state["termination"] = "repair_budget"
        return "end"

    return "continue"


def self_correct_and_execute(
    code: str,
    competition_name: str,
    corrector: CorrectorClient,
    execution: ExecutionConfig,
    recorder: Optional[RunRecorder] = None,
    n_repair: int = N_REPAIR,
) -> CorrectionState:
    """Run one candidate program through the full self-correction loop.

    Deep-learning programs that train for more than one epoch are first rewritten
    to a single epoch, so that a pipeline defect surfaces in minutes; the
    original epoch count is restored once the smoke test passes.
    """
    normalised = enforce_data_path(code, competition_name)
    if normalised != code:
        normalised = (
            f"# Task data is located in '{data_root(competition_name)}'\n" + normalised
        )

    is_dl_code, original_epochs, smoke_test_code = detect_dl_code_and_epochs(normalised)

    if is_dl_code and original_epochs and original_epochs > 1 and smoke_test_code:
        code_to_run = smoke_test_code
        testing_phase = True
        LOGGER.info(
            "Deep-learning program detected (epochs=%d); smoke-testing with epochs=1",
            original_epochs,
        )
    else:
        code_to_run = normalised
        testing_phase = False
        LOGGER.info(
            "Running directly (deep learning: %s, epochs: %s)", is_dl_code, original_epochs
        )

    initial_state: CorrectionState = {
        "original_code": normalised,
        "current_code": code_to_run,
        "corrections": [],
        "correction_count": 0,
        "error_messages": [],
        "status": "needs_correction",
        "is_dl_code": is_dl_code,
        "original_epochs": original_epochs,
        "testing_phase": testing_phase,
        "test_success": False,
        "test_execution_time": None,
        "competition_name": competition_name,
        "dl_testing_timeout": execution.dl_testing_timeout,
        "normal_timeout": execution.normal_timeout,
        "final_success": False,
        "timeout_count": 0,
        "n_repair": n_repair,
        "deadlock": False,
        "termination": "repair_budget",
        "last_similarity": 0.0,
    }

    graph = CorrectionLoop(corrector, execution, recorder).build()
    result = graph.invoke(initial_state)

    if not result["current_code"] or len(result["current_code"].strip()) < MIN_USABLE_CODE_LENGTH:
        LOGGER.warning("Final program looks truncated; falling back to the original")
        result["current_code"] = normalised

    return result
