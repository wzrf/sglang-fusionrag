# FusionRAG Unified Pipeline Handoff

## Snapshot
- Date: `2026-04-30`
- Local branch: `fusionrag_unified_pipeline`
- Local HEAD: `e5c8242cb7518df7c089ad6050133ad12cdc5f5f`
- Main local repo: `/Users/hming/code/sglang-fusionrag`

## Remote Worktrees on sa1
1. User original tree (do not modify for this task):
- `/mnt/data/shm/sglang-fusionrag`
- Branch: `use_hi_cache_qwen`
- Contains unrelated/dirty work.

2. Formal integration tree used by this task:
- `/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`
- Branch: `fusionrag_unified_pipeline`
- Synced from local via `rsync`.

## What Was Implemented
- New schema parser/validator: `python/sglang/srt/fusionrag_params.py`
- Request-local planning and guards: `python/sglang/srt/fusionrag_plan.py`
- Scheduler/request integration updates:
  - `python/sglang/srt/managers/tokenizer_manager.py`
  - `python/sglang/srt/managers/schedule_batch.py`
  - `python/sglang/srt/managers/schedule_policy.py`
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- Cache/memory path updates:
  - `python/sglang/srt/mem_cache/base_prefix_cache.py`
  - `python/sglang/srt/mem_cache/common.py`
  - `python/sglang/srt/mem_cache/fusionrag_cache.py`
  - `python/sglang/srt/mem_cache/memory_pool.py`
- Forward/logprob path updates:
  - `python/sglang/srt/model_executor/forward_batch_info.py`
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
  - `python/sglang/srt/layers/logits_processor.py`
- Misc runtime blockers fixed:
  - `python/sglang/srt/models/deepseek_v2_main.py`

## Tests Added
- `python/sglang/test/test_fusionrag_params.py`
- `python/sglang/test/test_fusionrag_runtime_smoke.py`
- Startup helper script:
  - `scripts/qwen_sglang_smoke.sh`

## Runtime Status (Important)
1. Progress achieved:
- Runtime smoke reaches real scheduler execution and request processing.
- Several hard failures were fixed (dtype mismatch, kv-gen save text assumption, missing `req` in plan-lookup path).

2. Current blocking scenario:
- `input logprob + cache hit` combination remains unsafe unless downgraded.
- Runtime downgrade path has been added in `schedule_batch.py` for input-logprob requests.

3. Last known issue fixed but not yet fully revalidated end-to-end:
- In downgrade branch, `Req.device` access was incorrect; changed to `tree_cache_fusionrag.device`.
- Needs final smoke re-run to confirm no regression.

## Repro Command for Next Agent
Use this exact environment (same as user setup style):

```bash
ssh sa1 '
  cd /mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline && \
  CUDA_VISIBLE_DEVICES=4 \
  FUSIONRAG_SMOKE_HICACHE_SIZE=32 \
  PYTHONPATH=/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/python \
  /mnt/data/shm/sglang-fusionrag/venv-kt/bin/python3.10 \
  python/sglang/test/test_fusionrag_runtime_smoke.py
'
```

## What Next Agent Should Do First
1. Re-run the runtime smoke command above and capture full logs.
2. Validate expected behavior:
- output-only logprob request: chunk lookup/hit path active and no assertion.
- input-logprob request: explicit downgrade and stable fallback reason, no assertion.
3. If assertion persists in scheduler output processing:
- prioritize correctness over hit-rate; keep downgrade broad enough to avoid incorrect input-logprob emission.
4. If smoke passes, finalize docs and minimal regression test notes.

## Known Non-Blocking Warnings
- `requests` dependency version warning in venv.
- `fusionrag_cache` may log unsupported old cache layout warnings when historical cache artifacts exist.

## Safety Notes
- Do not modify `/mnt/data/shm/sglang-fusionrag` directly for this task.
- Continue on `/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`.
- Avoid destructive git operations; repository is intentionally split to protect user’s dirty tree.
