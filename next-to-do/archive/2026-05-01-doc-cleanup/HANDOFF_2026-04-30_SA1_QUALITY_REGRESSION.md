# Handoff: SA1 Quality Regression (2026-04-30)

## 1) Current User-Visible Problem
File: `/mnt/data/shm/jybigdata/sources/iter_rag/results/sglang_test_rate_0.15.json`

Observed:
- 13 samples, `correct_sglang=3` (low quality).
- Typical failures are not simple truncation: they are noisy long-form rewrites, entity drift, and answer-off-topic outputs.
- Example patterns:
  - Query asks direct fact, output expands unrelated passage details.
  - "Julia's House" case shows hallucinated/shifted entity string.

Interpretation:
- This is a behavior/semantics regression relative to prior stable pipeline runs.
- Not a single prompt-only issue; likely residual runtime alignment issues in FusionRAG recompute/cache path under full pipeline load.

## 2) What Was Fixed in This Round

### A. Recompute crash root-fix chain
- `schedule_batch.py`
  - synchronized `recompute_idx` and `recompute_fill_idx` lifecycle.
  - removed stale recompute carry-over usage in compute-index build.
- `fusionrag_plan.py`
  - strict recompute policy:
    - map-only from chunk-hit spans,
    - disallow prefix-area recompute,
    - remove prefix-space fallback.
- `forward_batch_info.py`
  - `extend_start_loc` aligned to actual compute stream length (`extend_all_compute_len`).
- `flashinfer_backend.py`
  - prefill/query boundary construction aligned with recompute compute-length semantics.
- `mem_cache/common.py`
  - `page_size==1` allocation rebuilt from `compute_positions` to avoid K/V write length mismatch when recompute overlaps uncached tail.

### B. Endpoint compatibility fixes for eval scripts
- `run_ds_sglang_test.sh`: endpoint default switched to `/generate`.
- `run_sglang_test_single.sh`: endpoint default switched to `/generate`.
- `rag/agent/sglang_kvcache.py`: normalize `/v1/completions` and `/v1/chat/completions` to `/generate`.

## 3) Functional Delta vs Previous Commits
Reference commits:
- `211d31a64`: strict recompute mapping + flashinfer index alignment.
- Earlier merge/debug commits around unified fusionrag pipeline.

New in this step (after `211d31a64`):
- Added `page_size==1` compute-position aligned KV allocation fix in `mem_cache/common.py`.
- Added explicit quality regression documentation and `next-to-do` reorganization.

What remains unresolved:
- End-to-end answer quality still below old framework baseline at `rate=0.15` dataset run.
- Need targeted verification for `page_size>1` branch alignment (still old path logic).

## 4) Recommended Next Debug Actions
1. Add per-request integrity logging in extend path:
   - `len(input_ids_compute)`, `len(positions)`, `len(out_cache_loc_slice)`, `len(qo segment)`.
2. Apply compute-position-aligned allocation logic to `page_size>1` branch as well.
3. Re-run same dataset and compare against pre-regression baseline with fixed seed and identical retrieval set.

## 5) next-to-do Folder Layout (after cleanup)
- `next-to-do/active/`: actively used scripts.
- `next-to-do/archive/2026-04-30/`: old plans/handoffs/backup scripts.
- `next-to-do/HANDOFF_2026-04-30_SA1_RECOMPUTE_FIX_SUMMARY.md`: recompute root-fix summary.
- `next-to-do/HANDOFF_2026-04-30_SA1_QUALITY_REGRESSION.md`: this quality-regression handoff.


## 6) Newly Found Crash Bug (2026-04-30 22:12:43)
Traceback site:
- `python/sglang/srt/mem_cache/fusionrag_cache.py:565`
- current code does `raise f"panic! not enough memories!"`

Observed runtime error:
- `TypeError: exceptions must derive from BaseException`

Impact:
- scheduler process aborts during chunk cache load-back eviction path.
- hides the real OOM/eviction condition behind wrong exception type.

Required fix (next step):
- replace string raise with proper exception object, e.g. `raise RuntimeError("panic! not enough memories!")`.
- include request/chunk context in message for diagnosis.
