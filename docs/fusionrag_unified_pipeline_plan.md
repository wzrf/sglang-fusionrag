# FusionRAG Unified Pipeline Plan

## Branch and Scope
- Working branch: `fusionrag_unified_pipeline`
- Base branch: `origin/merge_hi_cache_qwen_prefix`
- Goal: align FusionRAG runtime behavior to the new `fusionrag_params` schema while reusing existing HiCache/FusionragCache infrastructure.

## What Is Completed
1. Request ingress schema freeze and normalization.
- Added `python/sglang/srt/fusionrag_params.py`.
- Tokenizer ingress now normalizes FusionRAG v2 schema in `python/sglang/srt/managers/tokenizer_manager.py`.

2. Request-local planning layer.
- Added `python/sglang/srt/fusionrag_plan.py` with `FusionRAGPlan`, save actions, recompute remap helpers, and runtime guards.
- `Req` lifecycle now carries and consumes request-local plan in `python/sglang/srt/managers/schedule_batch.py`.

3. Descriptor-based chunk lookup path.
- `fusionrag_cache.match_prefix()` now supports plan-driven lookup by `doc_id/chunk_id/doc_hash/cache_variant` in `python/sglang/srt/mem_cache/fusionrag_cache.py`.
- `MatchResult` carries matched chunk plan in `python/sglang/srt/mem_cache/base_prefix_cache.py`.

4. Runtime mapping and writeback fixes for recompute/compute positions.
- Sparse writeback support and compute-position mapping in `python/sglang/srt/mem_cache/common.py`.
- `prepare_for_extend` path updated to split compute length vs alloc length in `python/sglang/srt/managers/schedule_batch.py`.

5. Logprob metadata semantic split.
- Added `extend_input_lens` flow through:
  - `python/sglang/srt/managers/schedule_batch.py`
  - `python/sglang/srt/model_executor/forward_batch_info.py`
  - `python/sglang/srt/layers/logits_processor.py`
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

6. KV-gen save path hardening.
- Multi-variant save support and metadata propagation in `python/sglang/srt/mem_cache/fusionrag_cache.py`.
- `deepseek_v2_main` duplicate registration workaround in `python/sglang/srt/models/deepseek_v2_main.py`.
- Hierarchical pool transfer wait fix in `python/sglang/srt/mem_cache/memory_pool.py`.

7. Tests and tools added.
- Planner/unit-style tests: `python/sglang/test/test_fusionrag_params.py`.
- Runtime smoke harness: `python/sglang/test/test_fusionrag_runtime_smoke.py`.
- Remote startup helper: `scripts/qwen_sglang_smoke.sh`.

## Current Known Problems
1. `input logprob + cache hit` is not end-to-end safe yet.
- Observed scheduler assertion in `add_input_logprob_return_values` due to missing logits for cache-hit tokens.
- Current direction is runtime downgrade for input-logprob requests in FusionRAG path.

2. Runtime downgrade branch needs one more validation pass.
- A recent fix replaced `Req.device` with `tree_cache_fusionrag.device` in the downgrade path, but full smoke re-run after this fix is still pending.

3. KV cache reload warns about legacy layout mismatch during runtime smoke.
- `fusionrag_cache.py` logs `unsupport layout detected ...` while loading historical cache files.
- This is non-fatal in current smoke path but indicates old cache artifacts/schema mismatch in cache dirs.

## Remaining Plan
1. Close runtime smoke loop (blocking).
- Re-run `python/sglang/test/test_fusionrag_runtime_smoke.py` on `sa1` formal worktree.
- Verify:
  - output-only logprob request can use chunk hit path.
  - input-logprob request is safely downgraded with explicit fallback reason.

2. Harden logprob/fallback semantics.
- Keep fallback reason explicit and stable in response metadata.
- Ensure no scheduler assertion in mixed cache-hit/logprob cases.

3. Clean up cache-layout compatibility behavior.
- Decide whether to ignore old cache files by namespace/versioned key, or add migration guard.

4. Finalize handoff acceptance checks.
- Runtime smoke green on `sa1`.
- Planner tests pass in the same environment.
- Document known unsupported combinations clearly.

## Verification Commands
- Local compile check:
  - `python3 -m py_compile python/sglang/srt/managers/schedule_batch.py`
  - `python3 -m py_compile python/sglang/srt/mem_cache/fusionrag_cache.py`
- Remote smoke (`sa1`):
  - `cd /mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`
  - `CUDA_VISIBLE_DEVICES=4 FUSIONRAG_SMOKE_HICACHE_SIZE=32 PYTHONPATH=/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/python /mnt/data/shm/sglang-fusionrag/venv-kt/bin/python3.10 python/sglang/test/test_fusionrag_runtime_smoke.py`
