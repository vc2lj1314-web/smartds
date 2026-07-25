#!/usr/bin/env bash
# Example experiment launcher.
#
# Credentials and endpoints come from .env; nothing sensitive is written here.
# Adjust the task list, the budgets and the node label, then run:
#
#     bash scripts/run_experiment.sh
#
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

python -m autods.cli \
  --competitions \
      house-prices-advanced-regression-techniques \
      demo-shopee-iet-competition \
  --prompts data/task_prompts.json \
  --n-meta 10 \
  --n-repair 5 \
  --n-meta 10 \
  --n-repair 5 \
  --max-episodes 1 \
  --cpu-limit 4 \
  --dl-testing-timeout 1800 \
  --normal-timeout 7200 \
  --node-id node-0 \
  --output-root runs
