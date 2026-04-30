from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.fusionrag_plan import build_compute_cache_indices
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import support_triton
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def _split_recompute_mapping(req, recompute_cache_index: torch.Tensor) -> tuple[list[int], list[int]]:
    recompute_fill_idx = list(getattr(req, "recompute_fill_idx", None) or [])
    recompute_n = int(recompute_cache_index.numel())
    if len(recompute_fill_idx) != recompute_n:
        logger.warning(
            "[FusionRAG] recompute_alignment_mismatch: rid=%s recompute_fill=%d recompute_cache=%d",
            req.rid,
            len(recompute_fill_idx),
            recompute_n,
        )

    n_pair = min(len(recompute_fill_idx), recompute_n)
    return (
        [int(pos) for pos in recompute_fill_idx[:n_pair]],
        [int(cache_idx) for cache_idx in recompute_cache_index[:n_pair].tolist()],
    )


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


@triton.jit
def write_req_to_token_pool_sparse_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    flat_req_pool_indices,  # [num_sparse]
    flat_positions,  # [num_sparse]
    sparse_values,  # [num_sparse]
    num_sparse,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_sparse

    req_pool_index = tl.load(flat_req_pool_indices + offset, mask=mask, other=0)
    position = tl.load(flat_positions + offset, mask=mask, other=0)
    value = tl.load(sparse_values + offset, mask=mask, other=0)

    tl.store(
        req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + position,
        value,
        mask=mask,
    )


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
    compute_positions: list[list[int]] | None = None,
):
    # Triton fast path does contiguous [prefix, extend] writes.
    # When compute_positions is provided, we need sparse writes and fall back to Python path.
    if compute_positions is None and support_triton(get_global_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            device=req_to_token_pool.device,
            dtype=torch.uint64,
        )
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        sparse_req_indices = []
        sparse_positions = []
        sparse_values = []
        use_sparse_triton = (
            compute_positions is not None
            and support_triton(get_global_server_args().attention_backend)
        )
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()
            pool_dtype = req_to_token_pool.req_to_token.dtype

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i].to(dtype=pool_dtype),
            )
            if compute_positions is None:
                req_to_token_pool.write(
                    (req_idx, slice(prefix_len, seq_len)),
                    out_cache_loc[pt : pt + extend_len].to(dtype=pool_dtype),
                )
            else:
                if use_sparse_triton:
                    positions_i = compute_positions[i]
                    sparse_req_indices.extend([req_idx] * len(positions_i))
                    sparse_positions.extend(positions_i)
                    sparse_values.append(out_cache_loc[pt : pt + extend_len].to(dtype=pool_dtype))
                else:
                    req_to_token_pool.write(
                        (
                            req_idx,
                            torch.tensor(
                                compute_positions[i],
                                dtype=torch.int64,
                                device=req_to_token_pool.device,
                            ),
                        ),
                        out_cache_loc[pt : pt + extend_len].to(dtype=pool_dtype),
                    )
            pt += extend_len

        if use_sparse_triton and sparse_positions:
            sparse_req_indices_tensor = torch.tensor(
                sparse_req_indices, dtype=torch.int64, device=req_to_token_pool.device
            )
            sparse_positions_tensor = torch.tensor(
                sparse_positions, dtype=torch.int64, device=req_to_token_pool.device
            )
            sparse_values_tensor = torch.cat(sparse_values, dim=0)
            num_sparse = sparse_positions_tensor.shape[0]
            write_req_to_token_pool_sparse_triton[(triton.cdiv(num_sparse, 512),)](
                req_to_token_pool.req_to_token,
                sparse_req_indices_tensor,
                sparse_positions_tensor,
                sparse_values_tensor,
                num_sparse,
                req_to_token_pool.req_to_token.shape[1],
            )


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    if (
        get_global_server_args().attention_backend != "ascend"
        and get_global_server_args().attention_backend != "torch_native"
    ):
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
        factor = (
            MAMBA_STATE_PER_REQ_PREFIX_CACHE
            if tree_cache.supports_mamba()
            else MAMBA_STATE_PER_REQ_NO_CACHE
        )
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def alloc_for_extend(
    batch: ScheduleBatch,
    recompute_cache_indices: list[torch.Tensor],
    input_ids_only_extend: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        req_pool_indices_device: request pool indices at a device tensor
        req_pool_indices: request pool indices as list
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors = [r.prefix_indices for r in batch.reqs]
    compute_positions = batch.compute_positions

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.alloc_extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache_hicache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if batch.tree_cache_hicache.page_size == 1:
        # Build out_cache_loc strictly aligned with compute_positions.
        # This avoids shape mismatch when recompute overlaps uncached tail positions.
        out_cache_loc_chunks = []
        for i, req in enumerate(batch.reqs):
            req_positions = compute_positions[i]
            compute_len = len(req_positions)
            recompute_cache_index = recompute_cache_indices[i].to(torch.int64)
            recompute_n = int(recompute_cache_index.numel())
            uncached_n = len(input_ids_only_extend[i])
            recompute_positions, recompute_cache_values = _split_recompute_mapping(
                req, recompute_cache_index
            )
            recompute_pos_set = set(recompute_positions)
            fresh_n = sum(1 for pos in req_positions if pos not in recompute_pos_set)
            fresh_slots = alloc_token_slots(batch.tree_cache_hicache, fresh_n).to(torch.int64)
            req_cache_indices_tensor = torch.tensor(
                build_compute_cache_indices(
                    compute_positions=req_positions,
                    recompute_positions=recompute_positions,
                    recompute_cache_indices=recompute_cache_values,
                    fresh_cache_indices=[int(slot) for slot in fresh_slots.tolist()],
                ),
                dtype=torch.int64,
                device=batch.device,
            )
            if req_cache_indices_tensor.numel() != compute_len:
                raise RuntimeError(
                    f"[FusionRAG] kv_alloc_len_mismatch: rid={req.rid} compute_len={compute_len} alloc_len={req_cache_indices_tensor.numel()}"
                )
            out_cache_loc_chunks.append(req_cache_indices_tensor)

            if recompute_n > 0:
                logger.debug(
                    "[FusionRAG] kv_alloc_map: rid=%s recompute_n=%d uncached_n=%d compute_len=%d fresh_n=%d",
                    req.rid,
                    recompute_n,
                    uncached_n,
                    compute_len,
                    fresh_n,
                )

        out_cache_loc = torch.cat(out_cache_loc_chunks) if out_cache_loc_chunks else torch.tensor([], dtype=torch.int64, device=batch.device)

        write_cache_indices(
            out_cache_loc,
            req_pool_indices_device,
            req_pool_indices_cpu,
            prefix_lens_device,
            prefix_lens_cpu,
            batch.seq_lens,
            batch.seq_lens_cpu,
            torch.tensor(batch.extend_lens, dtype=torch.int64, device=batch.device),
            torch.tensor(batch.extend_lens, dtype=torch.int64),
            prefix_tensors,
            batch.req_to_token_pool,
            compute_positions=compute_positions,
        )

    else:
        # Paged allocation - allocate only fresh compute slots, then stitch
        # them back with recompute cache indices in compute_positions order.
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        fresh_lens = []
        recompute_mappings = []
        for i, req in enumerate(batch.reqs):
            recompute_positions, recompute_cache_values = _split_recompute_mapping(
                req, recompute_cache_indices[i].to(torch.int64)
            )
            recompute_mappings.append((recompute_positions, recompute_cache_values))
            recompute_pos_set = set(recompute_positions)
            fresh_lens.append(
                sum(1 for pos in compute_positions[i] if pos not in recompute_pos_set)
            )

        fresh_lens_cpu = torch.tensor(fresh_lens, dtype=torch.int64)
        fresh_lens_device = fresh_lens_cpu.to(batch.device, non_blocking=True)
        fresh_seq_lens_cpu = prefix_lens_cpu + fresh_lens_cpu
        fresh_seq_lens_device = prefix_lens_device + fresh_lens_device

        fresh_out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache_hicache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=fresh_seq_lens_device,
            seq_lens_cpu=fresh_seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=sum(fresh_lens),
        )
        out_cache_loc_chunks = []
        pt = 0
        for i, req in enumerate(batch.reqs):
            fresh_n = fresh_lens[i]
            req_positions = compute_positions[i]
            fresh_cache_indices = fresh_out_cache_loc[pt : pt + fresh_n].to(torch.int64)
            recompute_positions, recompute_cache_values = recompute_mappings[i]
            req_cache_indices_tensor = torch.tensor(
                build_compute_cache_indices(
                    compute_positions=req_positions,
                    recompute_positions=recompute_positions,
                    recompute_cache_indices=recompute_cache_values,
                    fresh_cache_indices=[int(slot) for slot in fresh_cache_indices.tolist()],
                ),
                dtype=torch.int64,
                device=batch.device,
            )
            out_cache_loc_chunks.append(req_cache_indices_tensor)
            pt += fresh_n
            if len(getattr(req, "recompute_idx", []) or []) > 0:
                logger.debug(
                    "[FusionRAG] kv_alloc_map: rid=%s recompute_cache_idx=%s fresh_n=%d out_cache_prefix=%s",
                    req.rid,
                    recompute_cache_values[:16],
                    fresh_n,
                    req_cache_indices_tensor[:16].tolist(),
                )
        out_cache_loc = (
            torch.cat(out_cache_loc_chunks)
            if out_cache_loc_chunks
            else torch.tensor([], dtype=torch.int64, device=batch.device)
        )

        # Write to req_to_token_pool
        write_cache_indices(
            out_cache_loc,
            req_pool_indices_device,
            req_pool_indices_cpu,
            prefix_lens_device,
            prefix_lens_cpu,
            batch.seq_lens,
            batch.seq_lens_cpu,
            torch.tensor(batch.extend_lens, dtype=torch.int64, device=batch.device),
            torch.tensor(batch.extend_lens, dtype=torch.int64),
            prefix_tensors,
            batch.req_to_token_pool,
            compute_positions=compute_positions,
        )

    return out_cache_loc, req_pool_indices_device, req_pool_indices


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    bs = batch.seq_lens.shape[0]

    if batch.tree_cache_hicache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache_hicache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        seq_lens_next = batch.seq_lens + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache_hicache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    if not req.is_kv_gen:
        tree_cache.cache_finished_req(req, is_insert=is_insert)

    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    if spec_algo is None:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
