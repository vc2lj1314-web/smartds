# AutoDS-Solver

Reference implementation of an autonomous data science agent that generates,
repairs and executes end-to-end machine learning programs for tabular, vision,
text and time-series tasks, while adapting its own decoding parameters between
attempts.

> **Anonymous artifact.** This repository accompanies a double-blind submission
> and deliberately contains no author names, affiliations, acknowledgements,
> internal host names, API keys, or links to non-anonymous repositories. Please
> do not open issues or pull requests that would deanonymise the authors.

---

## 1. What the system does

Given a natural-language description of a data science task and a directory of
raw data, the agent produces a scored submission without human intervention:

1. **Generate.** The policy model under evaluation writes a complete Python
   program, emitting its reasoning inside `<think>` tags and the program inside
   `<answer>` tags.
2. **Extract.** A cascade of parsers recovers an executable program from the
   response, and records whether the model honoured the output contract.
3. **Smoke-test.** Deep-learning programs are first rewritten to a single epoch,
   so a pipeline defect surfaces in minutes instead of hours. The original epoch
   count is restored once the smoke test passes.
4. **Self-correct.** A LangGraph loop runs the program under a CPU and
   wall-clock budget and, on failure, asks a *frozen auxiliary model* for a
   repair. The loop stops on success, on a repeated timeout, on a stalled repair
   loop, or when the repair budget is exhausted.
5. **Submit and score.** A produced submission file is sent to the leaderboard
   and its score is read back.
6. **Adapt.** A controller scores the attempt, decomposes the reward, and moves
   the sampling temperature and top-p for the next attempt.

Steps 1–6 form one **attempt**; a bounded sequence of attempts at one task is an
**episode**; a sequence of episodes over a task list is an **experiment**.
Parameter knowledge is carried across episodes, so the search is cumulative.

### The adaptive decoding controller

The controller is the component the paper's ablations target. Its central claim
is that *how* an attempt failed should determine which way the decoding
parameters move:

| Outcome of the attempt | Interpretation | Move |
| --- | --- | --- |
| No program could be extracted | Sampling is too conservative; the response collapsed into prose | Temperature and top-p **up**, with escalating step sizes on repeated failures |
| Program ran, score below target | Room to explore | **Up**, scaled non-linearly by the gap to the target |
| Program crashed | Sampling is too exploratory | **Down**, scaled by the severity of the failure |

Two stabilisers prevent the search from degenerating. Step sizes are damped as a
parameter approaches the edge of its per-episode window, so the search does not
saturate at a boundary. And the temperature is never pushed below a value that
has already produced an executable program in the same episode.

The reward is decomposed into four terms:

```
total = alpha * r_generation + beta * r_execution + delta * r_efficiency
```

`r_performance` — the leaderboard score — is computed and logged but carries
**weight zero** in the control signal. It is deliberately excluded: letting the
evaluation metric steer the decoding parameters within an episode would tune the
harness against the metric it is measured by. The term is retained in the record
because it is used post hoc, to select the episode's best configuration and for
the analyses reported in the paper.

---

## 2. Repository layout

```
autods-solver/
├── autods/
│   ├── config.py              # every constant, threshold and config dataclass
│   ├── state.py               # typed state of the correction loop
│   ├── prompts.py             # prompt-set selector
│   ├── prompts_zh.py          # the prompt set used for the reported results
│   ├── prompts_en.py          # English translation, for readability
│   ├── prompt_builders.py     # prompt assembly from templates + run state
│   ├── llm_clients.py         # policy model client, auxiliary repair client
│   ├── code_extraction.py     # recovering a program from a free-form response
│   ├── code_analysis.py       # syntax checks, DL detection, error taxonomy
│   ├── execution.py           # sandboxed execution under CPU/time budgets
│   ├── resources.py           # thread limits and CPU affinity
│   ├── similarity.py          # stalled-repair-loop detection
│   ├── correction_graph.py    # the LangGraph self-correction state machine
│   ├── parameter_learning.py  # the adaptive decoding controller
│   ├── param_history.py       # cross-episode parameter memory
│   ├── leaderboard.py         # submission and score parsing
│   ├── workspace.py           # per-episode artifact directories
│   ├── telemetry.py           # structured run record
│   ├── episode.py             # the attempt loop for one task
│   ├── experiment.py          # the outer loop over episodes and tasks
│   ├── logging_utils.py       # logging setup
│   └── cli.py                 # command-line entry point
├── scripts/run_experiment.sh  # example launcher
├── data/task_prompts.example.json
├── requirements.txt
└── .env.example
```

Reading order for a reviewer: `parameter_learning.py` (the contribution), then
`correction_graph.py` (the repair loop), then `episode.py` (how the two are
composed).

---

## 3. Installation

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The error-similarity check downloads a small multilingual sentence encoder on
first use. To run fully offline, pre-download it and export:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

---

## 4. Configuration

Copy `.env.example` to `.env` and fill it in. No credential is ever read from
source code.

`python -m autods.cli` loads `.env` itself, from the current working directory,
before it reads a single `*_API_*` variable — so it works whether you invoke it
directly or through `scripts/run_experiment.sh`. A variable already present in
the environment (e.g. exported on the command line) always wins over the file.
Point at a different file with `--env-file path/to/file`, or pass `--env-file ''`
to load nothing and rely purely on the environment.

If a run logs `No corrector API key found` or falls back to the model name
`local-policy-model`, `.env` was not found or a variable inside it was not
picked up — check that you are running from the directory containing `.env`,
or pass `--env-file` explicitly.

| Variable | Meaning |
| --- | --- |
| `GENERATOR_API_BASE` | OpenAI-compatible endpoint serving the policy model |
| `GENERATOR_MODEL` | Name the endpoint exposes the policy model under |
| `GENERATOR_API_KEY` | Usually `EMPTY` for a local server |
| `CORRECTOR_API_URL` | Chat-completions endpoint of the auxiliary repair model |
| `CORRECTOR_MODEL` | Auxiliary repair model name |
| `CORRECTOR_API_KEY` | Credential for that endpoint |

Leaderboard credentials are read by the competition platform's own CLI from its
standard config file, not from this repository.

### Prompt language

Two complete prompt sets ship with the repository.

| Set | File | Role |
| --- | --- | --- |
| `zh` (default) | `autods/prompts_zh.py` | **The experiment of record.** The exact wording used to produce the reported results. |
| `en` | `autods/prompts_en.py` | A faithful translation, so a reader who does not read Chinese can follow what the agent asks of each model. |

They are **not interchangeable**. The evaluated policy model was
instruction-tuned on Chinese reasoning traces, so its behaviour is conditioned
on the exact strings in `prompts_zh.py`. Switching sets changes the system under
measurement -- most visibly the rate at which the model honours the `<answer>`
output contract, which feeds directly into the decoding-parameter controller.

Select with `--prompt-language {zh,en}` or `AUTODS_PROMPT_LANGUAGE`. Running with
`en` logs a warning and should be read as a deliberate ablation on prompt
language, not as a reproduction of the reported numbers.

Both sets expose the same template names; `autods/prompts.py` validates this at
import time and is the only module the pipeline reads templates through, so the
selection is made once at start-up and takes effect everywhere.

The Chinese templates are reproduced verbatim from the research script. The only
mechanical change is that values interpolated at call time became named
`str.format` placeholders; the rendered text is byte-identical to what the
original f-strings produced. Please do not reword them: every edit breaks the
correspondence between this repository and the reported results.

### Task data and prompts

The generated program is always told to read from `../input/<task-name>/`, so
lay the data out as:

```
<working directory>/          # you run the agent here
../input/<task-name>/         # train.csv, test.csv, images/, ...
```

Task descriptions live in a JSON file, one object per task
(see `data/task_prompts.example.json`):

```json
[{"competition_name": "<task-name>", "prompt": "<natural-language description>"}]
```

Reference scores and optimisation directions live in `COMPETITION_TARGETS` in
`autods/config.py`. Add a new task by adding one entry there and one entry in
the prompt file.

---

## 5. Correspondence with Algorithm 1

The code implements Algorithm 1 directly. Every symbol in the algorithm has a
named counterpart:

| Algorithm 1 | Code | Notes |
| --- | --- | --- |
| `for t = 1 to N_meta` | `Episode.run` meta-step loop | `--n-meta`, default 10 |
| `for k = 0 to N_repair` | `correction_graph.self_correct_and_execute` | `--n-repair`, default 5; one execution plus at most `N_repair` repairs |
| `BuildState(H, D)` | `sate.build_state` | returns `MetaState` = Eq. 3 |
| `S_t = {S_gap, H_t, S_mode, S_sim}` | `MetaState.gap / .history / .mode / .sim` | |
| `SATE(S_t, D, H) -> (a_t, Q_t, theta_t)` | `sate.select_action` (a_t) + `prompt_builders` (Q_t) + `ParameterController.update` (theta_t) | split so the decoding controller's reward history stays out of action selection |
| `M_DS(Q_t; theta_t)` | `llm_clients.GeneratorClient.complete` | the policy model under evaluation |
| `LocalRepair(...)` | `Episode._local_repair` (meta-level, line 10) and `CorrectionLoop._repair` (inner, line 39) | both call the frozen auxiliary model |
| `FastTest(C)` | epochs=1 smoke test in `code_analysis.detect_dl_code_and_epochs` | trivially passes for non-iterative programs |
| `ExecuteEnv(C, full)` | `execution.run_program` | |
| `ErrorSim(Log(E), H) > tau_sim` | `similarity.error_sim` in `route_after_step` | `TAU_SIM`, default 0.8 |
| `Better`, `MeetTarget` | `param_history.is_better`, `ParameterController.score_error_ratio` | metric-direction aware |
| `C*`, `P_best` | `Episode.best_temperature / .best_performance` | persist across all `N_meta` meta-steps |

### Budget accounting

One task, worst case:

```
policy-model calls   <= N_meta                  = 10
executions           <= N_meta * (N_repair + 1) = 60
LocalRepair calls    <= N_meta * (N_repair + 1) = 60
```

The policy model is not called on a meta-step where `a_t = repair`, so the
reasoning-call count is a strict bound. This matches the observed averages in
the paper (7 reasoning calls, 19 modification calls, 26 total).

### `--max-episodes` is not part of Algorithm 1

Algorithm 1 describes **one** trajectory of `N_meta` meta-steps. `--max-episodes`
runs the whole trajectory again, carrying the best decoding parameters forward
through `best_historical_parameters.json`. It exists for the cross-run
parameter study and for variance estimates, and defaults to 1.

`--max-episodes 2 --n-meta 5` is **not** equivalent to `--n-meta 10`. Each episode
restarts the trajectory: fresh history `H`, fresh `C*`, fresh controller, and a
second `InitialPrompt` generation. Two independent 5-step trajectories explore
less than one 10-step trajectory, because no meta-step after the fifth can
condition on what the first five learned. To reproduce the paper, use the
default `--n-meta 10 --max-episodes 1`.

---

## 6. Running

```bash
python -m autods.cli \
    --competitions house-prices-advanced-regression-techniques \
    --prompts data/task_prompts.json \
    --n-meta 10 \
    --n-repair 5 \
    --output-root runs
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--no-submit` | Run the whole pipeline without touching the leaderboard |
| `--cpu-limit N` | Thread budget for each generated program (default 4) |
| `--dl-testing-timeout S` | Budget for the epochs=1 smoke test (default 1800 s) |
| `--normal-timeout S` | Budget for an ordinary or full training run (default 7200 s) |
| `--initial-temperature` / `--initial-top-p` | Override the starting point |
| `--config-adjustment K` | Offset the starting point, so parallel nodes explore different regions |
| `--n-meta N` | `N_meta`, meta-steps per task (default 10, the paper's value) |
| `--n-repair N` | `N_repair`, repairs inside one meta-step (default 5) |
| `--max-episodes N` | Independent episodes per task; not part of Algorithm 1 (default 1) |
| `--prompt-language {zh,en}` | Prompt set; `zh` is the default and the experiment of record |
| `--node-id LABEL` | Opaque label recorded with every result |
| `--cuda-device D` | Value for `CUDA_VISIBLE_DEVICES` |

Use `--no-submit` for a dry run: it exercises generation, extraction, execution
and the controller, and only skips the leaderboard round trip.

---

## 7. Outputs

```
runs/
├── run-log_<tag>.txt                     # full execution log
├── run-record_<tag>.json                 # per-attempt telemetry, flushed after every task
├── best_historical_parameters.json       # cross-episode parameter memory
├── submission_<task>_<tag>_attempt_<n>_<time>.csv
└── <task>_<tag>/
    ├── initial_response_attempt_1.txt    # unmodified model response
    ├── extracted_code_attempt_1.py       # program as extracted
    ├── corrected_code_attempt_1.py       # program that actually ran
    └── parameter_log.json                # parameters, rewards, best configuration
```

`run-record_<tag>.json` is the input to the token-efficiency and attempt-count
analyses; `parameter_log.json` is the input to the controller ablations.

Every artifact carries the episode tag, so nothing is overwritten when the same
task is run repeatedly, and any result can be traced back to the episode and
attempt that produced it.

---

## 8. Reproducibility notes

- **Stochasticity.** The pipeline is intentionally stochastic: the controller
  searches over decoding parameters, so a single run is not a point estimate.
  Reported results are aggregated over repeated episodes, and comparisons use a
  paired non-parametric test rather than mean differences.
- **The auxiliary model is part of the harness.** It runs at a fixed low
  temperature, with fixed prompts, identically for every configuration, so it
  cannot confound a comparison between policy models. Its token usage is
  accounted separately (`correction_tokens`) from the policy model's
  (`inference_tokens`).
- **The prompts are the ones that were run.** `prompts_zh.py` is verbatim; a
  template-level check confirms every string is byte-identical to the research
  script. The English set is documentation, not the experiment.
- **Budgets are enforced, not advisory.** Every generated program gets the same
  CPU and wall-clock budget. A program is counted as successful only if it exits
  cleanly *and* writes a submission file whose modification time moved; exiting
  with status 0 is not sufficient.
- **Programs that hide their submission write inside `try/except` are rejected
  before execution**, because such a program can exit cleanly while producing
  nothing.

### Differences from the internal research script

This repository is a modular rewrite of a single-file research script. The
control flow, reward decomposition, thresholds and prompts are unchanged. The
following defects were corrected during the rewrite; each is documented at the
site of the fix.

| Fix | Location |
| --- | --- |
| Data-path normalisation is now a single idempotent regular-expression pass. The previous chain of independent `str.replace` calls re-rewrote its own output and could produce a doubled path such as `../../input/<task>/<task>/`. | `correction_graph.enforce_data_path` |
| The leaderboard score regex now matches any float. The previous pattern only matched values below 1, silently dropping the score of every regression task whose metric is a raw error. | `leaderboard.parse_metrics` |
| Timeout detection keys on a machine-readable sentinel emitted by the runner. The previous free-text keyword list included markers as loose as `epochs=1`, which misclassified ordinary tracebacks as timeouts and aborted the repair loop early. | `code_analysis.is_timeout_error` |
| The two near-duplicate execution paths, which disagreed on how stale a pre-existing submission file had to be (60 s versus 1 s), were unified behind one function with a single documented tolerance. | `execution.run_program` |
| A dead branch testing for a hard-coded "40 minutes" message that the runner never emitted, together with two unused locals, was removed. | `correction_graph.CorrectionLoop.correct_code` |
| The failure-pattern index is now populated; the method that built it was previously never called. It remains observational and is never read back by the controller. | `parameter_learning._record_failure_pattern` |
| The refinement loop performs at most `N_repair` executions and at most `N_repair` LocalRepair calls per meta-step. Algorithm 1's `for k = 0 to N_repair` implies one further execution: the candidate produced by the last repair is executed there, whereas here it is discarded. The repair-call budget the paper reports (`N_meta * N_repair = 50`) is identical either way, and execution counts are not reported anywhere in the paper. Behaviour is preserved from the research script. | `correction_graph.route_after_step` |
| The CLI loads `.env` itself, from the current working directory, before any config reads the environment. Previously only `scripts/run_experiment.sh` sourced `.env`; a direct `python -m autods.cli ...` invocation saw an empty environment and every credential and model name silently fell back to its default. | `cli.load_dotenv`, called at the top of `cli.main` |
| The auxiliary repair model's system prompt now receives the real data path. In the original it sat in a plain (non-f) string literal, so the model was shown the uninterpolated text `'../input/{competition_name}/'`. | `prompts_zh.CODE_CORRECTION_SYSTEM_PROMPT` |
| Module-level mutable globals (the run record, the controller instance, the credentials) became explicit objects passed through the call graph, so two experiments can run in one process. | throughout |

Two behaviours were **kept** even though they look surprising, because changing
them would change the experimental setup; both are documented in place:

- The reasoning trace fed to the optimisation prompt is the one from the *first*
  attempt, not the most recent one (`episode.Episode.run`).
- The performance reward term carries weight zero
  (`parameter_learning`, module docstring).

---

## 9. Extending the benchmark

To add a task:

1. Add `"<task-name>": (<reference score>, "maximize" | "minimize")` to
   `COMPETITION_TARGETS` in `autods/config.py`.
2. Add `{"competition_name": "<task-name>", "prompt": "..."}` to the prompt file.
3. Place the data under `../input/<task-name>/`.
4. Pass `--competitions <task-name>`.

The task family (`nlp_tasks`, `vision_tasks`, `time_series`, `tabular_data`) is
inferred from the task name and prompt by keyword, and only determines the
starting decoding parameters; add keywords to `PROBLEM_TYPE_KEYWORDS` if a new
task is misclassified.

---

## 10. Licence

Released for peer review. A licence will be attached to the public release upon
acceptance.
