import pytest

from sglang.srt.fusionrag_params import (
    FusionRAGValidationError,
    normalize_fusionrag_params,
)
from sglang.srt.fusionrag_plan import (
    build_compute_cache_indices,
    build_req_to_token_row,
    build_fusionrag_plan,
    compute_extend_logprob_pruned_lens,
    compute_request_local_recompute_indices,
    disable_recompute_for_return_logprob,
    select_prefix_compatible_chunk_hits,
)


def test_normalize_generate_chunk_plan_merges_recompute():
    params = {
        "enable": True,
        "mode": "generate",
        "chunk_plan": [
            {
                "doc_id": "doc-1",
                "chunk_id": "doc-1-c0",
                "start_token": 3,
                "end_token": 8,
                "recompute_token_idx_local": [0, 2, 2],
            }
        ],
        "recompute_token_idx": [1, 5],
    }

    normalized = normalize_fusionrag_params(params, input_ids_len=16)

    assert normalized["cache_policy"]["use_radix_cache"] is True
    assert normalized["chunk_plan"][0]["recompute_token_idx_local"] == [0, 2]
    assert normalized["recompute_token_idx"] == [1, 3, 5]
    assert normalized["_fusionrag_schema_version"] == 2


def test_normalize_kv_gen_requires_kv_gen_block():
    params = {
        "enable": True,
        "mode": "kv_gen",
    }

    try:
        normalize_fusionrag_params(params, input_ids_len=8)
        assert False, "expected FusionRAGValidationError"
    except FusionRAGValidationError as exc:
        assert exc.code == "FRG006"


def test_normalize_rejects_overlapping_chunk_ranges():
    params = {
        "enable": True,
        "chunk_plan": [
            {"start_token": 0, "end_token": 4},
            {"start_token": 3, "end_token": 6},
        ],
    }

    try:
        normalize_fusionrag_params(params, input_ids_len=8)
        assert False, "expected FusionRAGValidationError"
    except FusionRAGValidationError as exc:
        assert exc.code == "FRG004"


def test_normalize_sorts_chunk_plan_before_overlap_validation():
    params = {
        "enable": True,
        "chunk_plan": [
            {"start_token": 4, "end_token": 6},
            {"start_token": 0, "end_token": 4},
        ],
        "recompute_token_idx": [],
    }

    normalized = normalize_fusionrag_params(params, input_ids_len=8)
    assert [chunk["start_token"] for chunk in normalized["chunk_plan"]] == [0, 4]


def test_normalize_rejects_chunk_ranges_exceeding_input_length():
    params = {
        "enable": True,
        "chunk_plan": [
            {"start_token": 0, "end_token": 9},
        ],
    }

    try:
        normalize_fusionrag_params(params, input_ids_len=8)
        assert False, "expected FusionRAGValidationError"
    except FusionRAGValidationError as exc:
        assert exc.code == "FRG003"


def test_build_plan_generates_kv_gen_save_actions():
    params = {
        "enable": True,
        "mode": "kv_gen",
        "cache_policy": {
            "save_to_radix_cache": False,
            "save_to_chunk_cache": True,
        },
        "chunk_plan": [
            {
                "doc_id": "doc-1",
                "chunk_id": "doc-1-c0",
                "cache_variant": "preprocess",
                "start_token": 4,
                "end_token": 10,
            }
        ],
        "kv_gen": {
            "target_doc_id": "doc-1",
            "target_chunk_id": "doc-1-c0",
            "save_variants": ["raw", "preprocess"],
            "strip_prefix": True,
        },
    }

    plan = build_fusionrag_plan(params, input_ids_len=12)

    assert plan is not None
    assert plan.is_kv_gen is True
    assert plan.kv_gen_prefix_len == 4
    assert [action.variant for action in plan.save_actions] == ["raw", "preprocess"]
    assert all(action.target == "chunk" for action in plan.save_actions)
    assert all(action.strip_prefix is True for action in plan.save_actions)


def test_compute_request_local_recompute_indices_with_prefix_and_chunk_hits():
    params = {
        "enable": True,
        "mode": "generate",
        "chunk_plan": [
            {
                "doc_id": "doc-1",
                "chunk_id": "doc-1-c0",
                "cache_variant": "preprocess",
                "start_token": 8,
                "end_token": 12,
                "recompute_token_idx_local": [1],
            },
            {
                "doc_id": "doc-2",
                "chunk_id": "doc-2-c0",
                "cache_variant": "raw",
                "start_token": 12,
                "end_token": 15,
                "recompute_token_idx_local": [],
            },
        ],
        "recompute_token_idx": [2, 9, 13],
    }

    plan = build_fusionrag_plan(params, input_ids_len=20)
    recompute_idx = compute_request_local_recompute_indices(
        plan,
        prefix_hicache_len=5,
        hit_chunk_plans=plan.chunk_plan,
        hit_chunk_token_lens=[4, 3],
        prefix_indices_len=12,
    )

    # Strict v2 policy drops prefix-space recompute and only keeps chunk-mapped positions.
    assert recompute_idx == [6, 10]


def test_select_prefix_compatible_chunk_hits_stops_on_gap():
    params = {
        "enable": True,
        "mode": "generate",
        "chunk_plan": [
            {"start_token": 5, "end_token": 8, "cache_variant": "raw"},
            {"start_token": 10, "end_token": 12, "cache_variant": "raw"},
        ],
    }

    plan = build_fusionrag_plan(params, input_ids_len=16)
    selected_plans, selected_lens, all_used = select_prefix_compatible_chunk_hits(
        prefix_hicache_len=5,
        hit_chunk_plans=plan.chunk_plan,
        hit_chunk_token_lens=[3, 2],
    )

    # v2 semantics: chunk spans are explicit overlays; gaps are allowed and handled by prefill.
    assert len(selected_plans) == 2
    assert selected_lens == [3, 2]
    assert all_used is True


def test_build_req_to_token_row_overwrites_real_compute_positions():
    row = build_req_to_token_row(
        seq_len=8,
        prefix_indices=[100, 101, 102, 103, 104],
        compute_positions=[2, 5, 6, 7],
        compute_cache_indices=[202, 305, 306, 307],
    )

    assert row == [100, 101, 202, 103, 104, 305, 306, 307]


def test_build_compute_cache_indices_deduplicates_overlap_between_recompute_and_tail():
    cache_indices = build_compute_cache_indices(
        compute_positions=[20, 21, 22, 23, 24, 25],
        recompute_positions=[20, 21, 22],
        recompute_cache_indices=[920, 921, 922],
        fresh_cache_indices=[1030, 1031, 1032],
    )

    assert cache_indices == [920, 921, 922, 1030, 1031, 1032]


def test_build_compute_cache_indices_rejects_wrong_fresh_slot_count():
    with pytest.raises(ValueError, match="fresh cache size mismatch"):
        build_compute_cache_indices(
            compute_positions=[5, 6, 7],
            recompute_positions=[5],
            recompute_cache_indices=[105],
            fresh_cache_indices=[206],
        )


def test_disable_recompute_for_return_logprob_downgrades_to_safe_path():
    recompute_idx, fallback_reason = disable_recompute_for_return_logprob(
        [1, 3, 5],
        return_logprob=True,
        extend_input_len=6,
        extend_logprob_start_len=2,
    )

    assert recompute_idx == []
    assert fallback_reason == "return_logprob_recompute_unsupported"


def test_disable_recompute_for_return_logprob_keeps_output_only_path():
    recompute_idx, fallback_reason = disable_recompute_for_return_logprob(
        [1, 3, 5],
        return_logprob=True,
        extend_input_len=6,
        extend_logprob_start_len=6,
    )

    assert recompute_idx == [1, 3, 5]
    assert fallback_reason is None


def test_compute_extend_logprob_pruned_lens_uses_input_lens_not_compute_lens():
    extend_return_logprob, pruned_lens = compute_extend_logprob_pruned_lens(
        [2],
        [2],
    )

    assert extend_return_logprob is False
    assert pruned_lens == [0]
