"""State passed between nodes of the correction graph."""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

CorrectionStatus = Literal["needs_correction", "completed", "timeout_error"]


class CorrectionState(TypedDict):
    """Mutable state of one self-correction loop over a single candidate program.

    Attributes:
        original_code: The program exactly as extracted from the generator.
        current_code: The program that will be executed on the next iteration.
        corrections: Every repaired version produced so far, oldest first.
        correction_count: Number of repair iterations performed.
        error_messages: Stderr / harness messages, oldest first.
        status: Terminal or continuation marker for the loop.
        is_dl_code: Whether the program was detected as deep-learning code.
        original_epochs: Epoch count found in the program before it was forced
            to 1 for the smoke test; ``None`` if no epoch setting was found.
        testing_phase: True while the program runs with epochs forced to 1.
        test_success: True once the epochs=1 smoke test has passed.
        test_execution_time: Wall-clock seconds of the smoke test, used to
            extrapolate the timeout for the full training run.
        competition_name: Task identifier, also the data sub-directory name.
        dl_testing_timeout: Timeout for the epochs=1 smoke test, in seconds.
        normal_timeout: Timeout for ordinary and full training runs, in seconds.
        final_success: True once a run has produced a fresh submission file.
        timeout_count: Number of timeouts observed in this loop.
        n_repair: Repair budget for this meta-step (N_repair in Algorithm 1).
        deadlock: True if the loop stopped because ErrorSim exceeded tau_sim.
        termination: Why the loop ended -- one of ``success``, ``deadlock``,
            ``timeout`` or ``repair_budget``. SATE reads this to build S_mode.
        last_similarity: The most recent ErrorSim value, recorded for analysis.
    """

    original_code: str
    current_code: str
    corrections: List[str]
    correction_count: int
    error_messages: List[str]
    status: CorrectionStatus
    is_dl_code: bool
    original_epochs: Optional[int]
    testing_phase: bool
    test_success: bool
    test_execution_time: Optional[float]
    competition_name: str
    dl_testing_timeout: int
    normal_timeout: int
    final_success: bool
    timeout_count: int
    n_repair: int
    deadlock: bool
    termination: str
    last_similarity: float
