"""Command-line entry point.

Everything the original script hard-coded in its ``__main__`` block -- endpoint,
credentials, model names, task list, timeouts, node label -- is a flag or an
environment variable here, so a run can be described completely by its command
line plus its ``.env``.

Example::

    python -m autods.cli \\
        --competitions house-prices-advanced-regression-techniques \\
        --prompts data/task_prompts.json \\
        --n-meta 10 --n-repair 5
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

from .config import (
    DEFAULT_CPU_LIMIT,
    N_META,
    N_REPAIR,
    DEFAULT_DL_TESTING_TIMEOUT,
    DEFAULT_NORMAL_TIMEOUT,
    BEST_HISTORICAL_PARAMS_FILE,
    CorrectorConfig,
    ExecutionConfig,
    ExperimentConfig,
    GeneratorConfig,
)
from .experiment import run_experiment
from .logging_utils import get_logger, setup_logging
from .prompts import active_language, set_prompt_language
from .resources import apply_process_cpu_limits, describe_cpu_limits

LOGGER = get_logger(__name__)

#: Default location of the environment file, relative to the current working
#: directory (i.e. wherever the command is invoked from).
DEFAULT_ENV_FILE = ".env"


def load_dotenv(path: str = DEFAULT_ENV_FILE) -> int:
    """Load ``KEY=VALUE`` pairs from a dotenv file into ``os.environ``.

    This exists so that ``.env`` takes effect regardless of how the CLI is
    invoked. ``scripts/run_experiment.sh`` sources ``.env`` itself before
    calling Python, but a direct ``python -m autods.cli ...`` invocation
    bypasses that script entirely -- in which case nothing has ever read the
    file, ``GeneratorConfig.from_env()`` and ``CorrectorConfig.from_env()`` see
    an empty environment, and every value silently falls back to its default
    (``local-policy-model``, an empty corrector key, ...). This loader closes
    that gap without adding a dependency on ``python-dotenv``.

    Values already present in the environment take precedence over the file,
    so a value exported on the command line before invocation
    (``GENERATOR_MODEL=foo python -m autods.cli ...``) still wins.

    Returns the number of variables set. Missing file: 0, silently.
    """
    if not os.path.isfile(path):
        return 0

    loaded = 0
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                LOGGER.warning("Skipping malformed line in %s: %r", path, raw_line.rstrip())
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]

            if key in os.environ:
                continue  # an explicit export before invocation wins
            os.environ[key] = value
            loaded += 1

    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autods",
        description="Run the AutoDS-Solver agent over a set of data science tasks.",
    )

    parser.add_argument(
        "--competitions", nargs="+", required=True,
        help="Task identifiers to run, matching the keys in COMPETITION_TARGETS.",
    )
    parser.add_argument(
        "--prompts", required=True,
        help="JSON file mapping each task identifier to its natural-language description.",
    )

    parser.add_argument("--n-meta", type=int, default=N_META,
                        help="N_meta: meta-steps per task (Algorithm 1 outer loop). This is "
                             "also the worst-case number of policy-model calls per task. "
                             f"Default {N_META}, the value used in the paper.")
    parser.add_argument("--n-repair", type=int, default=N_REPAIR,
                        help="N_repair: LocalRepair attempts allowed inside one meta-step "
                             f"(Algorithm 1 inner loop). Default {N_REPAIR}.")
    parser.add_argument("--max-episodes", type=int, default=1,
                        help="Independent episodes per task. Algorithm 1 describes one "
                             "trajectory of N_meta meta-steps, so the default is 1. Values "
                             "above 1 restart the trajectory with a fresh history, warm-starting "
                             "the decoding parameters from the best configuration found so far; "
                             "they do NOT compose into a longer trajectory.")

    parser.add_argument("--generator-base-url", default=None,
                        help="OpenAI-compatible endpoint of the policy model "
                             "(default: $GENERATOR_API_BASE).")
    parser.add_argument("--generator-model", default=None,
                        help="Policy model name (default: $GENERATOR_MODEL).")
    parser.add_argument("--corrector-model", default=None,
                        help="Auxiliary repair model name (default: $CORRECTOR_MODEL).")

    parser.add_argument("--cpu-limit", type=int, default=DEFAULT_CPU_LIMIT,
                        help="Thread budget for each generated program.")
    parser.add_argument("--dl-testing-timeout", type=int, default=DEFAULT_DL_TESTING_TIMEOUT,
                        help="Seconds allowed for the epochs=1 smoke test.")
    parser.add_argument("--normal-timeout", type=int, default=DEFAULT_NORMAL_TIMEOUT,
                        help="Seconds allowed for an ordinary or full training run.")
    parser.add_argument("--no-cpu-pinning", action="store_true",
                        help="Do not pin the process to a fixed set of cores.")

    parser.add_argument("--initial-temperature", type=float, default=None,
                        help="Override the starting temperature.")
    parser.add_argument("--initial-top-p", type=float, default=None,
                        help="Override the starting top-p.")
    parser.add_argument("--config-adjustment", type=int, default=0,
                        help="Integer offset applied to the starting parameters, so that "
                             "parallel nodes explore different regions.")

    parser.add_argument("--node-id", default="node-0",
                        help="Opaque label recorded with every result.")
    parser.add_argument("--cuda-device", default=None,
                        help="Value for CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--output-root", default=".",
                        help="Directory that receives episode artifacts and run records.")
    parser.add_argument("--history-file", default=BEST_HISTORICAL_PARAMS_FILE,
                        help="JSON file holding the cross-episode parameter history.")
    parser.add_argument("--no-submit", action="store_true",
                        help="Run the pipeline without submitting to the leaderboard.")
    parser.add_argument("--prompt-language", choices=("zh", "en"), default=None,
                        help="Prompt set to use. 'zh' is the wording that produced the "
                             "reported results and is the default; 'en' is a translation "
                             "kept for readability and constitutes a different experiment. "
                             "Defaults to $AUTODS_PROMPT_LANGUAGE, else 'zh'.")
    parser.add_argument("--log-file", default=None,
                        help="Log file path (default: run-log_<timestamp>.txt).")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE,
                        help="Dotenv file to load before reading any *_API_* variable "
                             "(default: .env in the current directory). Values already "
                             "set in the environment take precedence over the file. "
                             "Pass --env-file '' to skip loading one.")

    return parser


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Turn parsed arguments and the environment into an experiment config."""
    generator = GeneratorConfig.from_env()
    if args.generator_base_url:
        generator.base_url = args.generator_base_url
    if args.generator_model:
        generator.model = args.generator_model

    corrector = CorrectorConfig.from_env()
    if args.corrector_model:
        corrector.model = args.corrector_model

    execution = ExecutionConfig(
        cpu_limit=args.cpu_limit,
        dl_testing_timeout=args.dl_testing_timeout,
        normal_timeout=args.normal_timeout,
        pin_cpu_affinity=not args.no_cpu_pinning,
    )

    return ExperimentConfig(
        competitions=args.competitions,
        prompts_path=args.prompts,
        generator=generator,
        corrector=corrector,
        execution=execution,
        n_meta=args.n_meta,
        n_repair=args.n_repair,
        max_episodes=args.max_episodes,
        node_id=args.node_id,
        config_adjustment=args.config_adjustment,
        cuda_device=args.cuda_device,
        initial_temperature=args.initial_temperature,
        initial_top_p=args.initial_top_p,
        output_root=args.output_root,
        submit_to_leaderboard=not args.no_submit,
        history_file=args.history_file,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Must happen before build_config(): GeneratorConfig.from_env() and
    # CorrectorConfig.from_env() read os.environ at call time, so anything in
    # the dotenv file has to land there first.
    if args.env_file:
        loaded = load_dotenv(args.env_file)
        if loaded:
            print(f"Loaded {loaded} variable(s) from {args.env_file}", file=sys.stderr)
        elif os.path.isfile(args.env_file):
            print(
                f"{args.env_file} exists but every variable in it was already set "
                "in the environment", file=sys.stderr,
            )
        else:
            print(
                f"No env file at {args.env_file}; relying on variables already "
                "in the environment (if any)", file=sys.stderr,
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_file or os.path.join(args.output_root, f"run-log_{timestamp}.txt")
    os.makedirs(args.output_root, exist_ok=True)
    setup_logging(log_path)
    LOGGER.info("Logging to %s", log_path)

    # Select the prompt set before anything reads a template.
    if args.prompt_language:
        set_prompt_language(args.prompt_language)
    LOGGER.info("Prompt set: %s", active_language())

    config = build_config(args)

    apply_process_cpu_limits(config.execution.cpu_limit, config.execution.pin_cpu_affinity)
    LOGGER.info("CPU limits:\n%s", describe_cpu_limits(config.execution.cpu_limit))

    if config.submit_to_leaderboard and not config.corrector.api_key:
        LOGGER.warning(
            "No corrector API key found; the repair stage will fail. "
            "Set CORRECTOR_API_KEY (see .env.example)."
        )

    try:
        run_experiment(config)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by the user")
        return 130
    except Exception as exc:
        LOGGER.exception("Experiment failed: %s", exc)
        return 1

    LOGGER.info("Experiment finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
