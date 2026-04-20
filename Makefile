MODEL := allenai/olmOCR-2-7B-1025-FP8
RERANKER_MODEL := Qwen/Qwen3-Reranker-0.6B

GPU_MEMORY_UTILIZATION ?= 0.7
EMBD_GPU_MEMORY_UTILIZATION ?= 0.1
RERANKER_GPU_MEMORY_UTILIZATION ?= 0.3
EMBD_MODEL ?= Qwen/Qwen3-Embedding-0.6B
EMBD_PORT ?= 8002
RERANKER_PORT ?= 8003
MAX_MODEL_LEN          ?= 8192
TENSOR_PARALLEL_SIZE   ?= 1
DATA_PARALLEL_SIZE     ?= 1
PORT                   ?= 8000
API_PORT               ?= 8889
SERVER                 ?= localhost
PRIME_RL_DIR           ?= prime-rl/

.PHONY: vllm-olmocr-serve
vllm-olmocr-serve:
	uv run vllm serve $(MODEL) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-model-len $(MAX_MODEL_LEN) \
		--tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
		--data-parallel-size $(DATA_PARALLEL_SIZE) \
		--max-num-batched_tokens 16384 \
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

.PHONY: vllm-reranker-serve
vllm-reranker-serve:
	uv run vllm serve $(RERANKER_MODEL) \
		--gpu-memory-utilization $(RERANKER_GPU_MEMORY_UTILIZATION) \
		--hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
		--port $(RERANKER_PORT) \
		--host $(SERVER)

.PHONY: start-server
start-server:
	uv run uvicorn tool_server:app --host 0.0.0.0 --reload --port $(API_PORT)

.PHONY: run-ocr
run-ocr:
	nohup $(MAKE) vllm-olmocr-serve > olmocr.log 2>&1 &

.PHONY: run-embd
run-embd:
	nohup $(MAKE) vllm-embd-serve > embd.log 2>&1 &

.PHONY: run-server
run-server:
	nohup $(MAKE) start-server > server.log 2>&1 &

.PHONY: run-reranker
run-reranker:
	nohup $(MAKE) vllm-reranker-serve > reranker.log 2>&1 &

.PHONY: test
test:
	uv run pytest tests/ -v

.PHONY: train
train:
	cd $(PRIME_RL_DIR) && bash $(CURDIR)/run_train.sh
