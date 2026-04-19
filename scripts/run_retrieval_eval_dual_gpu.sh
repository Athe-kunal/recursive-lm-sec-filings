#!/usr/bin/env bash
set -euo pipefail

# Dynamic dual-GPU retrieval evaluation runner.
# Workflow:
#   1) Run BM25 baseline first.
#   2) Run dense model evaluations (dense + hybrid) with up to two concurrent jobs.
#      Jobs are assigned to whichever GPU becomes free first.

: "${EVAL_SCRIPT:=eval_retrieval.py}"
: "${EVAL_DATASET:=data/validation.parquet}"
: "${EVAL_TOP_K:=10}"
: "${WANDB_ENABLED:=true}"
: "${WANDB_PROJECT:=sec-filings-retrieval-eval}"
: "${WANDB_RUN_PREFIX:=dual-gpu-eval}"

: "${GPU_LIST:=2 3}"
: "${DENSE_MODELS:=Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-4B Qwen/Qwen3-Embedding-8B vespa-engine/colbert-medium BAAI/bge-base-en-v1.5}"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${message}"
}

normalize_model_name() {
  local model="$1"
  local normalized="${model//\//-}"
  normalized="${normalized//./-}"
  echo "${normalized}"
}

run_eval_mode() {
  local model="$1"
  local retrieval_mode="$2"
  local run_name="$3"
  local gpu_id="$4"
  local extra_args="$5"

  log "Starting eval with ${model=} ${retrieval_mode=} ${run_name=} ${gpu_id=}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    uv run python "${EVAL_SCRIPT}" \
      --dataset "${EVAL_DATASET}" \
      --embedding-model "${model}" \
      --retrieval-mode "${retrieval_mode}" \
      --top-k "${EVAL_TOP_K}" \
      --wandb-enabled "${WANDB_ENABLED}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-run-name "${run_name}" \
      --wandb-metric-group "${retrieval_mode}" \
      ${extra_args}
  log "Finished eval with ${model=} ${retrieval_mode=} ${run_name=} ${gpu_id=}"
}

run_dense_eval() {
  local model="$1"
  local run_name="$2"
  local gpu_id="$3"
  run_eval_mode "${model}" "dense" "${run_name}" "${gpu_id}" ""
}

run_hybrid_eval() {
  local model="$1"
  local run_name="$2"
  local gpu_id="$3"
  local extra_args="--fusion-method rrf --hybrid-sparse-retriever bm25 --hybrid-dense-retriever embedding"
  run_eval_mode "${model}" "hybrid" "${run_name}" "${gpu_id}" "${extra_args}"
}

run_model_pipeline() {
  local model="$1"
  local gpu_id="$2"
  local model_slug
  model_slug="$(normalize_model_name "${model}")"

  run_dense_eval "${model}" "${WANDB_RUN_PREFIX}-${model_slug}-dense" "${gpu_id}"
  run_hybrid_eval "${model}" "${WANDB_RUN_PREFIX}-${model_slug}-hybrid" "${gpu_id}"
}

run_bm25_baseline() {
  local run_name="${WANDB_RUN_PREFIX}-bm25"
  local retrieval_mode="bm25"

  log "Running BM25 baseline with ${run_name=} ${retrieval_mode=}"
  uv run python "${EVAL_SCRIPT}" \
    --dataset "${EVAL_DATASET}" \
    --embedding-model "bm25" \
    --retrieval-mode "${retrieval_mode}" \
    --top-k "${EVAL_TOP_K}" \
    --wandb-enabled "${WANDB_ENABLED}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${run_name}" \
    --wandb-metric-group "${retrieval_mode}"
}

find_finished_pid() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${pid}"
      return 0
    fi
  done
  return 1
}

remove_active_pid() {
  local target_pid="$1"
  local updated=()
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if [[ "${pid}" != "${target_pid}" ]]; then
      updated+=("${pid}")
    fi
  done
  ACTIVE_PIDS=("${updated[@]}")
}

acquire_free_gpu() {
  local idx
  for idx in "${!GPU_IDS[@]}"; do
    if [[ "${GPU_BUSY[idx]}" == "0" ]]; then
      GPU_BUSY[idx]="1"
      echo "${GPU_IDS[idx]}"
      return 0
    fi
  done
  return 1
}

mark_gpu_free() {
  local gpu_id="$1"
  local idx
  for idx in "${!GPU_IDS[@]}"; do
    if [[ "${GPU_IDS[idx]}" == "${gpu_id}" ]]; then
      GPU_BUSY[idx]="0"
      return 0
    fi
  done
}

launch_model_job() {
  local model="$1"
  local gpu_id="$2"

  run_model_pipeline "${model}" "${gpu_id}" &
  local pid=$!
  ACTIVE_PIDS+=("${pid}")
  PID_TO_GPU["${pid}"]="${gpu_id}"
  PID_TO_MODEL["${pid}"]="${model}"
  log "Launched job with ${pid=} ${model=} ${gpu_id=}"
}

reap_one_job() {
  wait -n

  local finished_pid
  finished_pid="$(find_finished_pid)"
  if [[ -z "${finished_pid}" ]]; then
    log "Warning: could not map finished PID to a running job."
    return 0
  fi

  local finished_gpu="${PID_TO_GPU[${finished_pid}]}"
  local finished_model="${PID_TO_MODEL[${finished_pid}]}"

  wait "${finished_pid}"
  mark_gpu_free "${finished_gpu}"
  remove_active_pid "${finished_pid}"
  unset PID_TO_GPU["${finished_pid}"]
  unset PID_TO_MODEL["${finished_pid}"]

  log "Completed job with ${finished_pid=} ${finished_model=} ${finished_gpu=}"
}

run_dense_model_queue() {
  local model

  for model in ${DENSE_MODELS}; do
    local gpu_id
    while ! gpu_id="$(acquire_free_gpu)"; do
      reap_one_job
    done
    launch_model_job "${model}" "${gpu_id}"
  done

  while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    reap_one_job
  done
}

initialize_gpu_state() {
  read -r -a GPU_IDS <<< "${GPU_LIST}"
  if [[ "${#GPU_IDS[@]}" -ne 2 ]]; then
    log "Expected exactly two GPU IDs in GPU_LIST, got ${GPU_LIST=}"
    exit 1
  fi

  GPU_BUSY=(0 0)
}

main() {
  log "Configuration: ${EVAL_SCRIPT=} ${EVAL_DATASET=} ${EVAL_TOP_K=} ${WANDB_PROJECT=} ${WANDB_RUN_PREFIX=}"
  log "Scheduling: ${GPU_LIST=} ${DENSE_MODELS=}"

  initialize_gpu_state
  run_bm25_baseline
  run_dense_model_queue

  log "All experiments completed successfully."
}

declare -a GPU_IDS
declare -a GPU_BUSY
declare -a ACTIVE_PIDS
ACTIVE_PIDS=()
declare -A PID_TO_GPU
declare -A PID_TO_MODEL

main "$@"
