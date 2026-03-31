MODEL := allenai/olmOCR-2-7B-1025-FP8

GPU_MEMORY_UTILIZATION ?= 0.97
EMBD_GPU_MEMORY_UTILIZATION ?= 0.1
EMBD_MODEL ?= Qwen/Qwen3-Embedding-0.6B
EMBD_PORT ?= 8888
MAX_MODEL_LEN          ?= 16384
TENSOR_PARALLEL_SIZE   ?= 2
DATA_PARALLEL_SIZE     ?= 1
PORT                   ?= 8000
API_PORT               ?= 8002
SERVER                 ?= localhost
PRIME_RL_DIR           ?= ../prime-rl

.PHONY: vllm-olmocr-serve
vllm-olmocr-serve:
	uv run vllm serve $(MODEL) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-model-len $(MAX_MODEL_LEN) \
		--tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
		--data-parallel-size $(DATA_PARALLEL_SIZE) \
		--max-num-batched_tokens 65536 \
		--max-num-seqs 8192 \
		--limit-mm-per-prompt '{"video": 0}' \
		--port $(PORT) \
		--host $(SERVER)

.PHONY: vllm-embd-serve
vllm-embd-serve:
	uv run vllm serve $(EMBD_MODEL) \
		--gpu-memory-utilization $(EMBD_GPU_MEMORY_UTILIZATION) \
		--runner pooling \
		--max-model-len 8192 \
		--port $(EMBD_PORT) \
		--host $(SERVER)

.PHONY: start-server
start-server:
	uv run uvicorn tool_server:app --host 0.0.0.0 --reload --port $(API_PORT)

.PHONY: test
test:
	uv run pytest tests/ -v

.PHONY: train
train:
	cd $(PRIME_RL_DIR) && bash $(CURDIR)/run_train.sh
