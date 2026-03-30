#!/usr/bin/env bash
set -euo pipefail
set -x

# PRIME-RL training wrapper for the SEC filings Verifiers environment.
#
# Setup reference:
#   https://github.com/PrimeIntellect-ai/prime-rl#setup
# Environment authoring reference:
#   https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/environments.md
#
# This script assumes PRIME-RL is installed in the current uv environment.
# It also assumes this repository is available as an editable package so
# PRIME-RL can import: rlm_sec.envs.finance_env:load_environment

: "${TRAIN_DATA:=data/train.parquet}"
: "${EVAL_DATA:=data/validation.parquet}"
: "${MAX_TURNS:=4}"
: "${TOPK:=3}"
: "${TIMEOUT:=30}"
: "${LOG_REQUESTS:=false}"
: "${TRAIN_CONFIG:=configs/debug/rl/train.toml}"

# PRIME-RL trainer entrypoint (from the official repository):
#   uv run trainer @ <train_config>
#
# We forward environment-specific overrides so that PRIME-RL loads this repo's
# verifiers ToolEnv factory.
uv run trainer @ "${TRAIN_CONFIG}" \
  env.module="rlm_sec.envs.finance_env:load_environment" \
  env.kwargs.train_paths="['${TRAIN_DATA}']" \
  env.kwargs.eval_paths="['${EVAL_DATA}']" \
  env.kwargs.max_turns="${MAX_TURNS}" \
  env.kwargs.topk="${TOPK}" \
  env.kwargs.timeout="${TIMEOUT}" \
  env.kwargs.log_requests="${LOG_REQUESTS}" \
  "$@"
