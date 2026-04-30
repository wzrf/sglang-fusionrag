#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline"
PYTHON_BIN="/mnt/data/shm/sglang-fusionrag/venv-kt/bin/python3.10"
MODEL_PATH="${FUSIONRAG_SMOKE_MODEL:-/mnt/data/models/Qwen2.5-0.5B-Instruct}"
PORT="${FUSIONRAG_SMOKE_PORT:-30012}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
HICACHE_SIZE="${FUSIONRAG_SMOKE_HICACHE_SIZE:-32}"

mkdir -p /mnt/data/tmp /mnt/data/triton_cache
export TMPDIR=/mnt/data/tmp
export TRITON_CACHE_DIR=/mnt/data/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/mnt/data/triton_cache/inductor
export SGLANG_DISABLED_MODEL_ARCHS="${SGLANG_DISABLED_MODEL_ARCHS:-deepseek_v2_main}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
"${PYTHON_BIN}" \
"${REPO_ROOT}/python/sglang/launch_server.py" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.7 \
  --chunked-prefill-size 2048 \
  --max-running-requests 8 \
  --max-total-tokens 4096 \
  --watchdog-timeout 3000 \
  --tensor-parallel-size 1 \
  --served-model-name fusionrag-smoke-qwen25-05b \
  --disable-shared-experts-fusion \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-size "${HICACHE_SIZE}" \
  --hicache-io-backend kernel \
  --hicache-write-policy write_back \
  --triton-attention-num-kv-splits 1 \
  --log-level info
