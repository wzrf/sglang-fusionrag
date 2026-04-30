from __future__ import annotations

from typing import Any, Dict, List, Optional


FUSIONRAG_DEFAULT_CACHE_POLICY = {
    "use_radix_cache": True,
    "use_chunk_cache": True,
    "save_to_radix_cache": True,
    "save_to_chunk_cache": False,
}

FUSIONRAG_V2_KEYS = {
    "enable",
    "mode",
    "cache_policy",
    "chunk_plan",
    "recompute_token_idx",
    "kv_gen",
}


class FusionRAGValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def is_fusionrag_v2_schema(params: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(params, dict):
        return False
    return any(key in params for key in FUSIONRAG_V2_KEYS)


def normalize_fusionrag_params(
    params: Optional[Dict[str, Any]],
    *,
    input_ids_len: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if params is None or not is_fusionrag_v2_schema(params):
        return params

    normalized = dict(params)
    enable = bool(normalized.get("enable", False))
    mode = normalized.get("mode", "generate")
    if mode not in ("generate", "kv_gen"):
        raise FusionRAGValidationError("FRG002", f"invalid mode: {mode}")

    cache_policy = dict(FUSIONRAG_DEFAULT_CACHE_POLICY)
    user_cache_policy = normalized.get("cache_policy") or {}
    if not isinstance(user_cache_policy, dict):
        raise FusionRAGValidationError(
            "FRG001", "cache_policy must be an object when fusionrag is enabled"
        )
    cache_policy.update(user_cache_policy)

    chunk_plan = _normalize_chunk_plan(
        normalized.get("chunk_plan", []),
        input_ids_len=input_ids_len,
    )
    recompute_idx = _normalize_global_recompute_idx(
        normalized.get("recompute_token_idx", []),
        input_ids_len=input_ids_len,
    )
    merged_recompute_idx = set(recompute_idx)
    for chunk in chunk_plan:
        merged_recompute_idx.update(
            chunk["start_token"] + idx
            for idx in chunk["recompute_token_idx_local"]
        )
    if input_ids_len is not None:
        merged_recompute_idx = {idx for idx in merged_recompute_idx if idx < input_ids_len}

    kv_gen = normalized.get("kv_gen")
    if mode == "kv_gen":
        kv_gen = _normalize_kv_gen(kv_gen)
    elif kv_gen is not None and not isinstance(kv_gen, dict):
        raise FusionRAGValidationError("FRG001", "kv_gen must be an object")

    normalized["enable"] = enable
    normalized["mode"] = mode
    normalized["cache_policy"] = cache_policy
    normalized["chunk_plan"] = chunk_plan
    normalized["recompute_token_idx"] = sorted(merged_recompute_idx)
    normalized["kv_gen"] = kv_gen
    normalized["_fusionrag_schema_version"] = 2
    return normalized


def _normalize_chunk_plan(
    chunk_plan: Any,
    *,
    input_ids_len: Optional[int],
) -> List[Dict[str, Any]]:
    if chunk_plan is None:
        return []
    if not isinstance(chunk_plan, list):
        raise FusionRAGValidationError("FRG001", "chunk_plan must be a list")

    raw_chunks = list(chunk_plan)
    raw_chunks.sort(
        key=lambda chunk: int(chunk.get("start_token", 0)) if isinstance(chunk, dict) else 0
    )

    normalized_chunks: List[Dict[str, Any]] = []
    last_end = -1
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            raise FusionRAGValidationError("FRG001", "chunk_plan item must be an object")

        start_token = _to_int(chunk.get("start_token"), "FRG003", "start_token is required")
        end_token = _to_int(chunk.get("end_token"), "FRG003", "end_token is required")
        if start_token >= end_token:
            raise FusionRAGValidationError(
                "FRG003",
                f"invalid chunk range [{start_token}, {end_token})",
            )
        if input_ids_len is not None and end_token > input_ids_len:
            raise FusionRAGValidationError(
                "FRG003",
                f"chunk range [{start_token}, {end_token}) exceeds input length {input_ids_len}",
            )
        if start_token < last_end:
            raise FusionRAGValidationError(
                "FRG004",
                f"overlapping chunk range [{start_token}, {end_token})",
            )

        span_len = end_token - start_token
        recompute_local = _normalize_local_recompute_idx(
            chunk.get("recompute_token_idx_local", []),
            span_len=span_len,
        )

        cache_variant = chunk.get("cache_variant", "raw")
        if cache_variant not in ("raw", "preprocess"):
            raise FusionRAGValidationError(
                "FRG001",
                f"unsupported cache_variant: {cache_variant}",
            )

        normalized_chunk = dict(chunk)
        normalized_chunk["start_token"] = start_token
        normalized_chunk["end_token"] = end_token
        normalized_chunk["cache_variant"] = cache_variant
        normalized_chunk["recompute_token_idx_local"] = recompute_local
        normalized_chunks.append(normalized_chunk)
        last_end = end_token

    return normalized_chunks


def _normalize_local_recompute_idx(value: Any, *, span_len: int) -> List[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise FusionRAGValidationError(
            "FRG001", "recompute_token_idx_local must be a list"
        )

    indices = sorted({int(idx) for idx in value})
    if any(idx < 0 or idx >= span_len for idx in indices):
        raise FusionRAGValidationError(
            "FRG005",
            f"recompute_token_idx_local out of range for span_len={span_len}",
        )
    return indices


def _normalize_global_recompute_idx(
    value: Any,
    *,
    input_ids_len: Optional[int],
) -> List[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise FusionRAGValidationError("FRG001", "recompute_token_idx must be a list")

    indices = sorted({int(idx) for idx in value})
    if any(idx < 0 for idx in indices):
        raise FusionRAGValidationError("FRG005", "recompute_token_idx must be >= 0")
    if input_ids_len is not None and any(idx >= input_ids_len for idx in indices):
        raise FusionRAGValidationError(
            "FRG005",
            f"recompute_token_idx must be < input length {input_ids_len}",
        )
    return indices


def _normalize_kv_gen(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise FusionRAGValidationError("FRG006", "kv_gen is required for mode=kv_gen")

    save_variants = value.get("save_variants", [])
    if not isinstance(save_variants, list) or len(save_variants) == 0:
        raise FusionRAGValidationError("FRG001", "kv_gen.save_variants must be a non-empty list")

    invalid_variants = [
        variant for variant in save_variants if variant not in ("raw", "preprocess")
    ]
    if invalid_variants:
        raise FusionRAGValidationError(
            "FRG001",
            f"unsupported save_variants: {invalid_variants}",
        )

    normalized = dict(value)
    normalized["save_variants"] = list(dict.fromkeys(save_variants))
    normalized["strip_prefix"] = bool(value.get("strip_prefix", False))
    return normalized


def _to_int(value: Any, code: str, message: str) -> int:
    if value is None:
        raise FusionRAGValidationError(code, message)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FusionRAGValidationError(code, message) from exc
