# Plan: Enable FusionRAG on Qwen2.5

## Goals
- Make Qwen2.5 follow the same FusionRAG flow used by DeepSeek (load cache → select recompute → set fusion_rag_indices → save cache).
- Ensure attention path actually consumes fusion_rag_indices (not just set).
- Keep scheduler and request plumbing unchanged.

## Steps
1) **Study DeepSeek FusionRAG flow**
   - Locate the load/save/recompute logic in `python/sglang/srt/models/deepseek_v2.py`.
   - Identify the exact data written into `forward_batch` and the KV cache APIs used.

2) **Add FusionRAG flow to Qwen2**
   - Implement the same fusionrag_params handling in `python/sglang/srt/models/qwen2.py`.
   - Add model-specific cache paths (based on model name) instead of DeepSeek hardcoding.

3) **Make attention honor fusion_rag_indices**
   - After review, recompute selection is already applied in `ScheduleBatch.prepare_for_extend()` by building `input_ids` from `all_compute_idx`. This means attention already computes only recompute+extend tokens, so no attention code changes are required for the initial port.

4) **Sanity checks & summary**
   - Confirm no scheduler changes required.
   - List touched files and the behavior changes.
