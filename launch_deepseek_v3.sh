#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/mnt/data/shm/sglang-fusionrag/venv-kt/bin/python3.10 \
/mnt/data/shm/sglang-fusionrag/python/sglang/launch_server.py \
--host 0.0.0.0 \
--port 30000 \
--model /mnt/data/models/DeepSeek-V3.2 \
--kt-weight-path /mnt/data/models/DeepSeek-V3.2 \
--kt-cpuinfer 100 \
--kt-threadpool-count 2 \
--kt-num-gpu-experts 0 \
--kt-method FP8 \
--kt-gpu-prefill-token-threshold 4096 \
--attention-backend triton \
--trust-remote-code \
--mem-fraction-static 0.8 \
--chunked-prefill-size 32768 \
--max-running-requests 16 \
--max-total-tokens 100000 \
--watchdog-timeout 3000 \
--tensor-parallel-size 4 \
--enable-p2p-check \
--served-model-name DeepSeek-V3.2 \
--disable-shared-experts-fusion \
--fp8-gemm-backend triton \
--disable-overlap-schedule \
--enable-hierarchical-cache \
--hicache-size 40 \
--hicache-io-backend kernel \
--hicache-write-policy write_back \
--triton-attention-num-kv-splits 1 \
--log-level info \
2>&1 | tee /tmp/sg_deepseek_v3.log
