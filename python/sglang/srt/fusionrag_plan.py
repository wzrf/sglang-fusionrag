from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.fusionrag_params import normalize_fusionrag_params

logger = logging.getLogger(__name__)


@dataclass
class FusionRAGChunkPlan:
    start_token: int
    end_token: int
    cache_variant: str
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    doc_hash: Optional[str] = None
    recompute_token_idx_local: List[int] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionRAGSaveAction:
    target: str
    variant: str
    strip_prefix: bool = False
    target_doc_id: Optional[str] = None
    target_chunk_id: Optional[str] = None
    target_doc_hash: Optional[str] = None


@dataclass
class FusionRAGPlan:
    enable: bool = False
    mode: str = "generate"
    cache_policy: Dict[str, Any] = field(default_factory=dict)
    chunk_plan: List[FusionRAGChunkPlan] = field(default_factory=list)
    recompute_token_idx: List[int] = field(default_factory=list)
    kv_gen: Optional[Dict[str, Any]] = None
    save_actions: List[FusionRAGSaveAction] = field(default_factory=list)
    request_token_len: int = 0
    schema_version: Optional[int] = None

    @property
    def is_kv_gen(self) -> bool:
        return self.enable and self.mode == "kv_gen"

    @property
    def primary_chunk_variant(self) -> str:
        if self.chunk_plan:
            return self.chunk_plan[0].cache_variant
        return "raw"

    @property
    def use_preprocess_cache(self) -> bool:
        return self.primary_chunk_variant == "preprocess"

    @property
    def kv_gen_prefix_len(self) -> int:
        if not self.is_kv_gen:
            return 0
        kv_gen = self.kv_gen or {}
        if not kv_gen.get("strip_prefix", False):
            return 0

        target_doc_id = kv_gen.get("target_doc_id")
        target_chunk_id = kv_gen.get("target_chunk_id")
        for chunk in self.chunk_plan:
            if target_chunk_id is not None and chunk.chunk_id != target_chunk_id:
                continue
            if target_doc_id is not None and chunk.doc_id != target_doc_id:
                continue
            return chunk.start_token

        if self.chunk_plan:
            return self.chunk_plan[0].start_token
        return 0


def build_fusionrag_plan(
    params: Optional[Dict[str, Any]],
    *,
    input_ids_len: int,
) -> Optional[FusionRAGPlan]:
    normalized = normalize_fusionrag_params(params, input_ids_len=input_ids_len)
    if normalized is None:
        return None

    chunk_plan = [
        FusionRAGChunkPlan(
            start_token=chunk["start_token"],
            end_token=chunk["end_token"],
            cache_variant=chunk["cache_variant"],
            doc_id=chunk.get("doc_id"),
            chunk_id=chunk.get("chunk_id"),
            doc_hash=chunk.get("doc_hash"),
            recompute_token_idx_local=list(chunk.get("recompute_token_idx_local", [])),
            raw=chunk,
        )
        for chunk in normalized.get("chunk_plan", [])
    ]

    plan = FusionRAGPlan(
        enable=bool(normalized.get("enable", False)),
        mode=normalized.get("mode", "generate"),
        cache_policy=dict(normalized.get("cache_policy", {})),
        chunk_plan=chunk_plan,
        recompute_token_idx=list(normalized.get("recompute_token_idx", [])),
        kv_gen=normalized.get("kv_gen"),
        request_token_len=input_ids_len,
        schema_version=normalized.get("_fusionrag_schema_version"),
    )
    plan.save_actions = _build_save_actions(plan)
    return plan


def _build_save_actions(plan: FusionRAGPlan) -> List[FusionRAGSaveAction]:
    if not plan.enable:
        return []

    actions: List[FusionRAGSaveAction] = []
    if plan.mode == "generate":
        if plan.cache_policy.get("save_to_radix_cache", True):
            actions.append(FusionRAGSaveAction(target="radix", variant="full"))
        if plan.cache_policy.get("save_to_chunk_cache", False):
            for chunk in plan.chunk_plan:
                actions.append(
                    FusionRAGSaveAction(
                        target="chunk",
                        variant=chunk.cache_variant,
                        target_doc_id=chunk.doc_id,
                        target_chunk_id=chunk.chunk_id,
                        target_doc_hash=chunk.doc_hash,
                    )
                )
        return actions

    kv_gen = plan.kv_gen or {}
    for variant in kv_gen.get("save_variants", []):
        actions.append(
            FusionRAGSaveAction(
                target="chunk",
                variant=variant,
                strip_prefix=bool(kv_gen.get("strip_prefix", False)),
                target_doc_id=kv_gen.get("target_doc_id"),
                target_chunk_id=kv_gen.get("target_chunk_id"),
                target_doc_hash=kv_gen.get("target_doc_hash"),
            )
        )
    if plan.cache_policy.get("save_to_radix_cache", False):
        actions.append(FusionRAGSaveAction(target="radix", variant="full"))
    return actions


def apply_legacy_fusionrag_fields(
    req: Any,
    params: Optional[Dict[str, Any]],
    plan: Optional[FusionRAGPlan],
) -> None:
    req.is_kv_gen = False
    req.kv_gen_prefix_len = 0
    req.prefix_prompt = ""
    req.save_preprocess_cache = False
    req.save_raw_cache = False
    req.use_preprocess_cache = False
    req.recompute_idx = []
    req.prompt_ids_list = []
    req.prefix_prompt_ids_list = []
    req.prefix_cache_ids = []

    if plan is not None and plan.schema_version == 2:
        req.is_kv_gen = plan.is_kv_gen
        req.use_preprocess_cache = plan.use_preprocess_cache
        req.recompute_idx = list(plan.recompute_token_idx)
        req.kv_gen_prefix_len = plan.kv_gen_prefix_len

        chunk_save_variants = [
            action.variant for action in plan.save_actions if action.target == "chunk"
        ]
        req.save_preprocess_cache = "preprocess" in chunk_save_variants
        req.save_raw_cache = "raw" in chunk_save_variants
        if req.is_kv_gen and req.save_preprocess_cache:
            req.use_preprocess_cache = False
        return

    if params is None:
        return

    req.is_kv_gen = params.get("save_cache", False)
    req.prefix_prompt = params.get("prefix_prompt", "")
    req.save_preprocess_cache = params.get("save_preprocess_cache", False)
    req.recompute_idx = params.get("recompute_idx", [])
    req.save_raw_cache = not req.save_preprocess_cache
    req.use_preprocess_cache = params.get("load_preprocess_cache", False)
    req.prompt_ids_list = params.get("prompt_ids_list", [])
    req.prefix_prompt_ids_list = params.get("prefix_prompt_ids_list", [])
    req.kv_gen_prefix_len = len(params.get("prefix_prompt_ids", []))
    req.prefix_cache_ids = params.get("prefix_cache_ids", [])
    if req.is_kv_gen and req.save_preprocess_cache:
        req.use_preprocess_cache = False


def compute_request_local_recompute_indices(
    plan: Optional[FusionRAGPlan],
    *,
    prefix_hicache_len: int,
    hit_chunk_plans: List[FusionRAGChunkPlan],
    hit_chunk_token_lens: List[int],
    prefix_indices_len: int,
    fallback_recompute_idx: Optional[List[int]] = None,
) -> List[int]:
    if plan is None or getattr(plan, "schema_version", None) != 2:
        return sorted(
            idx for idx in (fallback_recompute_idx or []) if idx < prefix_indices_len
        )

    # Strict policy:
    # 1) recompute must be mapped from chunk hits;
    # 2) recompute is never allowed in radix-prefix area [0, prefix_hicache_len).
    local_recompute = set()
    mapped_global_recompute = set()
    dropped_prefix_global = []

    fusionrag_offset = prefix_hicache_len
    for chunk_plan, loaded_len in zip(hit_chunk_plans, hit_chunk_token_lens):
        span_start = chunk_plan.start_token
        span_end = min(chunk_plan.end_token, span_start + loaded_len)

        for global_idx in plan.recompute_token_idx:
            if span_start <= global_idx < span_end:
                if global_idx < prefix_hicache_len:
                    dropped_prefix_global.append(global_idx)
                    continue
                local_recompute.add(fusionrag_offset + (global_idx - span_start))
                mapped_global_recompute.add(global_idx)

        for local_idx in chunk_plan.recompute_token_idx_local:
            if 0 <= local_idx < loaded_len:
                global_idx = span_start + local_idx
                if global_idx < prefix_hicache_len:
                    dropped_prefix_global.append(global_idx)
                    continue
                local_recompute.add(fusionrag_offset + local_idx)

        fusionrag_offset += loaded_len

    if dropped_prefix_global:
        dropped_prefix_global = sorted(set(dropped_prefix_global))
        logger.debug(
            "[FusionRAG] drop_prefix_recompute_idx_strict: count=%d sample=%s prefix_hicache_len=%d",
            len(dropped_prefix_global),
            dropped_prefix_global[:16],
            prefix_hicache_len,
        )

    dropped_unmapped = [idx for idx in plan.recompute_token_idx if idx not in mapped_global_recompute]
    if dropped_unmapped:
        logger.debug(
            "[FusionRAG] drop_unmapped_recompute_idx: count=%d sample=%s prefix_hicache_len=%d chunk_hits=%d",
            len(dropped_unmapped),
            dropped_unmapped[:16],
            prefix_hicache_len,
            len(hit_chunk_plans),
        )

    # No prefix-space fallback here by design.
    return sorted(idx for idx in local_recompute if idx < prefix_indices_len)


def select_prefix_compatible_chunk_hits(
    *,
    prefix_hicache_len: int,
    hit_chunk_plans: List[FusionRAGChunkPlan],
    hit_chunk_token_lens: List[int],
) -> Tuple[List[FusionRAGChunkPlan], List[int], bool]:
    # v2 semantics (full.md): chunk hits are overlays on explicit `[start_token, end_token)`
    # spans provided by the client. They do NOT need to be contiguous with the radix/hicache
    # prefix hit length, and gaps between chunk spans are allowed (they will be prefetched).
    #
    # Keep this function for backward compatibility with call sites: now it simply returns
    # all hits as usable, and the scheduler will decide prefill spans for gaps.
    del prefix_hicache_len
    return list(hit_chunk_plans), list(hit_chunk_token_lens), True


def disable_recompute_for_return_logprob(
    recompute_idx: List[int],
    *,
    return_logprob: bool,
    extend_input_len: Optional[int] = None,
    extend_logprob_start_len: Optional[int] = None,
) -> Tuple[List[int], Optional[str]]:
    if not return_logprob or not recompute_idx:
        return list(recompute_idx), None
    # If this extend batch does not compute any logprobs, recompute is safe.
    if (
        extend_input_len is not None
        and extend_logprob_start_len is not None
        and extend_logprob_start_len >= extend_input_len
    ):
        return list(recompute_idx), None

    # For "output-only" logprobs (API `logprob_start_len == -1`), the scheduler will internally
    # set `extend_logprob_start_len` to `extend_input_len` for the prefill batch, so we won't hit
    # this branch. If we do, be conservative and only disable recompute when it overlaps with the
    # range that needs logprob computation inside the extend batch.
    if extend_input_len is None or extend_logprob_start_len is None:
        return [], "return_logprob_recompute_unsupported"

    if any(idx >= extend_logprob_start_len for idx in recompute_idx):
        return [], "return_logprob_recompute_unsupported"
    return list(recompute_idx), None


def compute_extend_logprob_pruned_lens(
    extend_input_lens: List[int],
    extend_logprob_start_lens: List[int],
) -> Tuple[bool, List[int]]:
    extend_return_logprob = False
    extend_logprob_pruned_lens = []
    for extend_input_len, start_len in zip(
        extend_input_lens,
        extend_logprob_start_lens,
    ):
        pruned_len = extend_input_len - start_len
        if pruned_len > 0:
            extend_return_logprob = True
        extend_logprob_pruned_lens.append(pruned_len)
    return extend_return_logprob, extend_logprob_pruned_lens


def build_req_to_token_row(
    *,
    seq_len: int,
    prefix_indices: List[int],
    compute_positions: List[int],
    compute_cache_indices: List[int],
) -> List[int]:
    if len(compute_positions) != len(compute_cache_indices):
        raise ValueError("compute_positions and compute_cache_indices must have the same length")
    if len(prefix_indices) > seq_len:
        raise ValueError("prefix_indices longer than seq_len")

    row = [0] * seq_len
    row[: len(prefix_indices)] = list(prefix_indices)

    for position, cache_index in zip(compute_positions, compute_cache_indices):
        if position < 0 or position >= seq_len:
            raise ValueError(f"compute position {position} out of range for seq_len={seq_len}")
        row[position] = cache_index

    return row


def build_compute_cache_indices(
    *,
    compute_positions: List[int],
    recompute_positions: List[int],
    recompute_cache_indices: List[int],
    fresh_cache_indices: List[int],
) -> List[int]:
    if len(recompute_positions) != len(recompute_cache_indices):
        raise ValueError("recompute positions and cache indices must have the same length")

    recompute_pos_to_cache: Dict[int, int] = {}
    for position, cache_index in zip(recompute_positions, recompute_cache_indices):
        previous = recompute_pos_to_cache.get(position)
        if previous is not None and previous != cache_index:
            raise ValueError(f"conflicting recompute cache index for position {position}")
        recompute_pos_to_cache[position] = cache_index

    fresh_needed = sum(1 for position in compute_positions if position not in recompute_pos_to_cache)
    if fresh_needed != len(fresh_cache_indices):
        raise ValueError(
            f"fresh cache size mismatch: expected {fresh_needed}, got {len(fresh_cache_indices)}"
        )

    ordered_cache_indices: List[int] = []
    fresh_pt = 0
    for position in compute_positions:
        cache_index = recompute_pos_to_cache.get(position)
        if cache_index is None:
            cache_index = fresh_cache_indices[fresh_pt]
            fresh_pt += 1
        ordered_cache_indices.append(cache_index)

    return ordered_cache_indices
