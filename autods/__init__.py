"""AutoDS-Solver: an autonomous data science agent with adaptive decoding control.

Public entry points:

* :func:`autods.experiment.run_experiment` -- run a full experiment.
* :func:`autods.episode.run_episode` -- run one episode over one task.
* :class:`autods.parameter_learning.ParameterController` -- the decoding-parameter
  controller, usable on its own.
"""

from .config import (  # noqa: F401
    CorrectorConfig,
    ExecutionConfig,
    ExperimentConfig,
    GeneratorConfig,
)
from .episode import EpisodeResult, run_episode  # noqa: F401
from .experiment import run_experiment  # noqa: F401
from .parameter_learning import ParameterController  # noqa: F401

__all__ = [
    "CorrectorConfig",
    "ExecutionConfig",
    "ExperimentConfig",
    "GeneratorConfig",
    "EpisodeResult",
    "ParameterController",
    "run_episode",
    "run_experiment",
]

__version__ = "1.0.0"
