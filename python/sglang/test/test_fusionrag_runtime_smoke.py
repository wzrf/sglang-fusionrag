import os
import socket
import sys
import uuid

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_request(engine, req):
    generator = engine.tokenizer_manager.generate_request(req, None)
    return engine.loop.run_until_complete(generator.__anext__())


def _unwrap_singleton(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _build_input_ids(tokenizer) -> list[int]:
    text = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho sigma tau upsilon"
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) < 10:
        raise RuntimeError(f"expected at least 10 input ids, got {len(input_ids)}")
    return input_ids[:10]


def main():
    if torch is None:
        print(
            "fusionrag runtime smoke skipped: missing dependency 'torch' "
            "(requires GPU runtime environment)",
            file=sys.stderr,
        )
        return

    os.environ.setdefault("SGLANG_DISABLED_MODEL_ARCHS", "deepseek_v2_main")
    model_path = os.environ.get(
        "FUSIONRAG_SMOKE_MODEL", "/mnt/data/models/Qwen2.5-0.5B-Instruct"
    )
    if not os.path.exists(model_path):
        print(
            f"fusionrag runtime smoke skipped: model path not found: {model_path}",
            file=sys.stderr,
        )
        return

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import GenerateReqInput
    port = int(os.environ.get("FUSIONRAG_SMOKE_PORT", _find_free_port()))
    served_model_name = os.environ.get(
        "FUSIONRAG_SMOKE_SERVED_MODEL", "fusionrag-smoke-qwen25-05b"
    )
    hicache_size = int(os.environ.get("FUSIONRAG_SMOKE_HICACHE_SIZE", "24"))
    print(
        {
            "model_path": model_path,
            "port": port,
            "served_model_name": served_model_name,
            "hicache_size": hicache_size,
        },
        flush=True,
    )

    engine = Engine(
        model_path=model_path,
        served_model_name=served_model_name,
        port=port,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.7,
        context_length=2048,
        max_running_requests=4,
        chunked_prefill_size=2048,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        enable_hierarchical_cache=True,
        hicache_size=hicache_size,
        hicache_write_policy="write_back",
        hicache_io_backend="kernel",
        attention_backend="triton",
        log_level="error",
    )

    try:
        tokenizer = engine.tokenizer_manager.tokenizer
        input_ids = _build_input_ids(tokenizer)
        chunk_end = 8
        rid_suffix = uuid.uuid4().hex[:8]

        kv_gen_req = GenerateReqInput(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
            fusionrag_params={
                "enable": True,
                "mode": "kv_gen",
                "cache_policy": {
                    "use_radix_cache": False,
                    "use_chunk_cache": True,
                    "save_to_radix_cache": False,
                    "save_to_chunk_cache": True,
                },
                "chunk_plan": [
                    {
                        "doc_id": f"doc-{rid_suffix}",
                        "chunk_id": f"chunk-{rid_suffix}",
                        "cache_variant": "raw",
                        "start_token": 0,
                        "end_token": chunk_end,
                    }
                ],
                "kv_gen": {
                    "target_doc_id": f"doc-{rid_suffix}",
                    "target_chunk_id": f"chunk-{rid_suffix}",
                    "save_variants": ["raw"],
                    "strip_prefix": False,
                },
            },
        )
        kv_gen_resp = _run_request(engine, kv_gen_req)

        output_only_req = GenerateReqInput(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
            return_logprob=True,
            logprob_start_len=-1,
            fusionrag_params={
                "enable": True,
                "mode": "generate",
                "cache_policy": {
                    "use_radix_cache": False,
                    "use_chunk_cache": True,
                    "save_to_radix_cache": False,
                    "save_to_chunk_cache": False,
                },
                "chunk_plan": [
                    {
                        "doc_id": f"doc-{rid_suffix}",
                        "chunk_id": f"chunk-{rid_suffix}",
                        "cache_variant": "raw",
                        "start_token": 0,
                        "end_token": chunk_end,
                    }
                ],
                "recompute_token_idx": [1, 5],
            },
        )
        output_only_resp = _run_request(engine, output_only_req)

        input_logprob_req = GenerateReqInput(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
            return_logprob=True,
            logprob_start_len=0,
            fusionrag_params={
                "enable": True,
                "mode": "generate",
                "cache_policy": {
                    "use_radix_cache": False,
                    "use_chunk_cache": True,
                    "save_to_radix_cache": False,
                    "save_to_chunk_cache": False,
                },
                "chunk_plan": [
                    {
                        "doc_id": f"doc-{rid_suffix}",
                        "chunk_id": f"chunk-{rid_suffix}",
                        "cache_variant": "raw",
                        "start_token": 0,
                        "end_token": chunk_end,
                    }
                ],
                "recompute_token_idx": [1, 5],
            },
        )
        input_logprob_resp = _run_request(engine, input_logprob_req)

        kv_meta = kv_gen_resp["meta_info"]
        out_meta = output_only_resp["meta_info"]
        in_meta = input_logprob_resp["meta_info"]

        print(
            {
                "kv_meta": kv_meta,
                "output_only_meta": out_meta,
                "input_logprob_meta": in_meta,
            },
            flush=True,
        )

        out_chunk_hit_tokens = _unwrap_singleton(out_meta["fusionrag_chunk_hit_tokens"])
        out_recompute_tokens = _unwrap_singleton(out_meta["fusionrag_recompute_tokens"])
        out_fallback_reason = _unwrap_singleton(out_meta["fusionrag_fallback_reason"])
        in_chunk_hit_tokens = _unwrap_singleton(in_meta["fusionrag_chunk_hit_tokens"])
        in_recompute_tokens = _unwrap_singleton(in_meta["fusionrag_recompute_tokens"])
        in_fallback_reason = _unwrap_singleton(in_meta["fusionrag_fallback_reason"])

        assert kv_meta["completion_tokens"] == 1
        assert out_chunk_hit_tokens >= chunk_end
        assert out_recompute_tokens == 2
        assert out_fallback_reason == ""
        assert len(out_meta["output_token_logprobs"]) == 1

        assert in_chunk_hit_tokens == 0
        assert in_recompute_tokens == 0
        assert in_fallback_reason == "input_logprob_cache_hit_unsupported"
        assert len(in_meta["input_token_logprobs"]) > 0

        print(
            {
                "kv_gen_completion_tokens": kv_meta["completion_tokens"],
                "output_only_chunk_hit_tokens": out_chunk_hit_tokens,
                "output_only_recompute_tokens": out_recompute_tokens,
                "output_only_fallback_reason": out_fallback_reason,
                "input_logprob_chunk_hit_tokens": in_chunk_hit_tokens,
                "input_logprob_recompute_tokens": in_recompute_tokens,
                "input_logprob_fallback_reason": in_fallback_reason,
            }
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fusionrag runtime smoke failed: {type(exc).__name__}: {exc}")
        raise
