# SA1 Recompute Root-Fix Summary (2026-04-30)

## Core Problems Fixed
1. Recompute stale-state mismatch:
- `recompute_idx` could be cleared while stale `recompute_fill_idx` remained, causing compute/write shape mismatch and crashes (`expected 4 got 1`).

2. Recompute coordinate fallback corruption:
- Global recompute indices were previously allowed to fallback into prefix-space when mapping failed, which could overwrite stable prefix area and corrupt semantics.

3. FlashInfer recompute path index consistency:
- Recompute path used real compute stream (`recompute + uncached`) but some boundary/index construction still followed raw extend length semantics.

## Code Changes
- `python/sglang/srt/managers/schedule_batch.py`
  - Keep `recompute_fill_idx` consistent with `recompute_idx` lifecycle.
  - Guard against stale recompute carry-over in `prepare_for_extend`.
  - Added structured logs for `extend_shape` and `recompute_map`.

- `python/sglang/srt/fusionrag_plan.py`
  - Enforced strict recompute policy:
    - recompute must come from mapped chunk-hit spans;
    - disallow recompute in radix-prefix region `[0, prefix_hicache_len)`;
    - remove prefix-space fallback behavior.
  - Added logs:
    - `drop_prefix_recompute_idx_strict`
    - `drop_unmapped_recompute_idx`

- `python/sglang/srt/model_executor/forward_batch_info.py`
  - `extend_start_loc` now uses actual compute stream lengths (`extend_all_compute_len`) instead of raw `extend_seq_lens`.

- `python/sglang/srt/layers/attention/flashinfer_backend.py`
  - In recompute path, align prefill/query boundary construction with `extend_all_compute_len` semantics.

## Validation Snapshot
- Crash path (`kvcache.cuh expected 4 got 1`) no longer reproduced.
- Strict policy behaves as designed:
  - recompute indices in prefix area are dropped;
  - non-prefix mapped recompute can execute (`recompute_n > 0`) with stable logs.

## Related Runner/Agent Endpoint Compatibility
- `run_ds_sglang_test.sh` and `run_sglang_test_single.sh` switched default endpoint to `/generate`.
- `rag/agent/sglang_kvcache.py` now normalizes endpoint suffixes (`/v1/completions`, `/v1/chat/completions`) to `/generate`.
