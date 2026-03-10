MODEL := allenai/olmOCR-2-7B-1025-FP8

GPU_MEMORY_UTILIZATION ?= 0.97
MAX_MODEL_LEN          ?= 16384
TENSOR_PARALLEL_SIZE   ?= 2
DATA_PARALLEL_SIZE     ?= 1
PORT                   ?= 8000
SERVER                 ?= localhost

.PHONY: vllm-olmocr-serve
vllm-olmocr-serve:
	uv run vllm serve $(MODEL) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-model-len $(MAX_MODEL_LEN) \
		--tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
		--data-parallel-size $(DATA_PARALLEL_SIZE) \
		--max-num-batched_tokens 65536 \
		--max-num-seqs 8192 \
		--port $(PORT) \
		--host $(SERVER)
