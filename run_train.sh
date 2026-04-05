#!/usr/bin/env bash
set -euo pipefail

# Prime RL training entrypoint for the SEC filings custom environment.
#
# Setup (from Prime RL README):
#   1) git clone https://github.com/PrimeIntellect-ai/prime-rl.git
#   2) cd prime-rl && uv sync --all-extras
#   3) Install this repo so Prime RL can import the environment:
#        uv pip install -e /workspace/recursive-lm-sec-filings
#
# This script assumes you run it from inside the prime-rl repository.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/rl.toml"

: "${RLM_SEC_TRAIN_DATA:=data/train.parquet}"
: "${RLM_SEC_EVAL_DATA:=data/validation.parquet}"
: "${RLM_SEC_TOPK:=3}"
: "${RLM_SEC_TIMEOUT:=30}"
: "${RLM_SEC_MAX_QA_TURNS:=4}"
: "${RLM_SEC_MAX_RANKING_TURNS:=1}"
: "${WANDB_PROJECT:=sec-filings-rl}"
: "${WANDB_NAME:=sec-filings-lora}"

if [[ "${RLM_SEC_TRAIN_DATA}" != /* ]]; then
  RLM_SEC_TRAIN_DATA="${SCRIPT_DIR}/${RLM_SEC_TRAIN_DATA}"
fi

if [[ "${RLM_SEC_EVAL_DATA}" != /* ]]; then
  RLM_SEC_EVAL_DATA="${SCRIPT_DIR}/${RLM_SEC_EVAL_DATA}"
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Run 'uv run wandb login' or export WANDB_API_KEY before training." >&2
fi

export RLM_SEC_TRAIN_DATA
export RLM_SEC_EVAL_DATA
export RLM_SEC_TOPK
export RLM_SEC_TIMEOUT
export RLM_SEC_MAX_QA_TURNS
export RLM_SEC_MAX_RANKING_TURNS

export WANDB_PROJECT
export WANDB_NAME

uv run rl @ "${CONFIG_PATH}" "$@"
