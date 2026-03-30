set -euxo pipefail

# Prime RL training entrypoint for the finance verifiers environment.
#
# Setup (from Prime RL README):
#   1) git clone https://github.com/PrimeIntellect-ai/prime-rl.git
#   2) cd prime-rl && uv sync --all-extras
#   3) Install this repo so Prime RL can import the environment:
#        uv pip install -e /workspace/recursive-lm-sec-filings
#
# This script assumes you run it from inside the prime-rl repository.

: "${RLM_SEC_TRAIN_DATA:=/workspace/recursive-lm-sec-filings/data/train.parquet}"
: "${RLM_SEC_EVAL_DATA:=/workspace/recursive-lm-sec-filings/data/validation.parquet}"
: "${RLM_SEC_TOPK:=3}"
: "${RLM_SEC_TIMEOUT:=30}"
: "${RLM_SEC_MAX_TURNS:=4}"

export RLM_SEC_TRAIN_DATA
export RLM_SEC_EVAL_DATA
export RLM_SEC_TOPK
export RLM_SEC_TIMEOUT
export RLM_SEC_MAX_TURNS

# Example debug run on Prime RL configs.
# Update environment import path in your train TOML to use:
#   rlm_sec.envs.prime_environment:load_environment
uv run trainer @ configs/debug/rl/train.toml "$@"
