from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional
import heapq
import bisect
import torch
import os, copy
from typing import Any
import hashlib

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.metrics.collector import StorageMetricsCollector
from sglang.srt.utils import bind_to_closest_numa_node_cuda

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

class HitCacheNode:
    def __init__(
        self,
        value: torch.Tensor,
        original_position: torch.Tensor,
        current_position: torch.Tensor,
    ):
        self.original_position: torch.Tensor = original_position
        self.current_position: torch.Tensor = current_position
        self.value: torch.Tensor = value

class ChunkNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.prefix_text: str = ""
        self.text_without_prefix: str = ""
        self.cache_prefix_token_len : int = 0
        self.values: Optional[List[torch.Tensor]] = None
        self.evictable_values: Optional[List[torch.Tensor]] = None
        self.key: Optional[List[int]] = None
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0 ##fixme: 按理说不需要，因为永远不会被从host释放
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority
        self.is_preprocess_cache = False ## false is raw cache, true is preprocess cache

        self.id = ChunkNode.counter if id is None else id
        ChunkNode.counter += 1

    @property
    def evicted(self):
        return len(self.values) == 0

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    # @lru_cache(maxsize=1)
    # def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
    #     if node is None or node.hash_value is None:
    #         return []
    #
    #     return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


class FusionragCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        if server_args.hicache_io_backend == "direct":
            # FIXME: move this logic into server_args parsing
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                )

        if not server_args.disable_hicache_numa_detect:
            bind_to_closest_numa_node_cuda()

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        if isinstance(self.kv_cache, MHATokenToKVPool): # QWEN用的是这个？
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        # TODO: support more timeout check functions

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=self.prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
        )
        if self.enable_storage_metrics:
            # TODO: support pp
            labels = {
                "storage_backend": server_args.hicache_storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
            }
            self.storage_metrics_collector = StorageMetricsCollector(labels=labels)

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.all_nodes: List[ChunkNode] = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.eviction_policy = "lru" ## mengyao_debug
        self.eviction_strategy: EvictionStrategy = LRUStrategy()

        cache_path_root = "/mnt/data3"
        # if not os.path.exists(cache_path_root):
        #     cache_path_root = "/mnt/data"
        self.cache_path = f"/mnt/data3/shm/fusionrag_tree_cache/DeepSeek-v3.2/raw_kv_cache"
        self.preprocess_cache_path = f"/mnt/data3/shm/fusionrag_tree_cache/DeepSeek-v3.2/preprocess_kv_cache"
        if os.environ.get("DEBUG", "0") != "0":
            self.cache_path = f"/mnt/data3/shm/fusionrag_tree_cache_DEBUG/DeepSeek-v3.2/raw_kv_cache"
            self.preprocess_cache_path = f"/mnt/data3/shm/fusionrag_tree_cache_DEBUG/DeepSeek-v3.2/preprocess_kv_cache"
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.preprocess_cache_path, exist_ok=True)

        super().__init__(params=params)
        self.load_all_from_ssd()

    def list_all_chunk_caches(self, use_preprocess_cache: bool):
        result = []
        if use_preprocess_cache:
            cache_path = self.preprocess_cache_path
        else:
            cache_path = self.cache_path
        for folder in os.listdir(cache_path):
            folder_path = os.path.join(cache_path, folder) ## folder is the md5
            if os.path.isdir(folder_path):
                metadata_path = os.path.join(folder_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        text_without_prefix = metadata.get("text", "")
                        cache_prefix_token_len = metadata.get("cache_prefix_token_len", "")
                        prefix_text = metadata.get("prefix_text", "")
                        result.append(
                            (text_without_prefix,
                             os.path.join(folder_path, f"{folder}.pt"),
                             cache_prefix_token_len,
                             prefix_text,
                             use_preprocess_cache)
                        )

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error reading {metadata_path}: {e}")

        return result

    def load_all_from_ssd(self):
        raw_chunk_caches = self.list_all_chunk_caches(use_preprocess_cache=False)
        preprocess_chunk_caches = self.list_all_chunk_caches(use_preprocess_cache=True)
        all_chunk_caches = []
        all_chunk_caches.extend(raw_chunk_caches)
        all_chunk_caches.extend(preprocess_chunk_caches)
        for all_chunk_cache in all_chunk_caches:
            text_without_prefix = all_chunk_cache[0]
            tensor_path = all_chunk_cache[1]
            cache_prefix_token_len = all_chunk_cache[2]
            prefix_text = all_chunk_cache[3]
            is_preprocess_cache = all_chunk_cache[4]
            chunk_tensor = torch.load(tensor_path, weights_only=True).to("cpu")
            prefetch_length = chunk_tensor.shape[1]
            try:
                if self.cache_controller.mem_pool_host.layer_num != chunk_tensor.shape[0]:
                    print(f"shape mismatch.")
                    continue
                host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
                if host_indices is None:
                    raise "failed to allocate host indices"
                self.cache_controller.mem_pool_host.set_from_indices(
                    host_indices,
                    chunk_tensor
                )
                node = ChunkNode()
                node.text_without_prefix = text_without_prefix
                node.prefix_text = prefix_text
                node.cache_prefix_token_len = cache_prefix_token_len
                node.is_preprocess_cache = is_preprocess_cache
                node.host_value = host_indices
                node.values = []
                self.all_nodes.append(node)
            except Exception as e:
                print(f"unsupport layout detected.")

        ## sort.
        self.all_nodes.sort(key=lambda n: len(n.text_without_prefix), reverse=True)


    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config JSON if provided
        extra_config = {}
        if storage_backend_extra_config:
            try:
                extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )


    def reset(self):
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def get_height(self, node):
        return 0

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    ##fixme: 推理的时候不用写回主存，因为kvcache会被污染，但是在计算kvcache的时候需要写回主存
    def write_backup(self, node: ChunkNode, write_back=False):
        host_indices = self.cache_controller.write(
            device_indices=node.values[0],
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.values[0],
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            assert len(node.host_value) > 0
            self.ongoing_write_through[node.id] = node
        else:
            return 0

        return len(host_indices)

    ## fixme: no need to write back to storage
    def write_backup_storage(self, node: ChunkNode):
        return
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            node.host_value, node.key, node.hash_value, prefix_keys
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _inc_hit_count(self, node: ChunkNode, chunked=False):
        # skip the hit count update for chunked requests
        ""

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        del self.ongoing_write_through[ack_id]
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                # self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            # for ack_id in ack_list:
            #     end_node = self.ongoing_load_back.pop(ack_id)
                # self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]


    def evictable_size(self):
        return self.evictable_size_

    ##fixme: evict的数据不需要存储到host，会污染kvcache
    def evict(self, num_tokens: int):
        print(f"we need to evict {num_tokens} tokens")
        raise f"panic! not enough memories!"
        ""

    def _evict_backuped(self, node: ChunkNode):
        ""

    def evict_host(self, num_tokens: int):
        """
        we skip this for now
        """


    def load_back(
        self, nodes_to_load: List[ChunkNode], mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor, list[torch.Tensor]]:
        # todo: more loading policies

        start_time = time.perf_counter()
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        last_hit_node = nodes_to_load[-1]
        ancester_node = nodes_to_load[0]

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            print(f"not enough HBM to load")
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        if device_indices is None:
            raise Exception(f"not enough HBM to load")
            # no sufficient GPU memory to load back KV caches
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        all_values = []
        for node in nodes_to_load:
            device_hit_index = device_indices[offset : offset + len(node.host_value)]
            node.values.append(device_hit_index)
            all_values.append(
                HitCacheNode(
                    value=device_hit_index,
                    current_position=torch.arange(offset, offset + len(node.host_value)).to(device_hit_index.device),
                    original_position=torch.arange(node.cache_prefix_token_len,
                                                   node.cache_prefix_token_len + len(node.host_value)).to(device_hit_index.device),
                )
            )
            offset += len(node.host_value)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        # return device_indices[:-1], all_values ## left one just for decode
        return device_indices, all_values  ## mengyao_debug hardcode

    def init_load_back_chunk(
        self,
        all_hit_nodes: List[ChunkNode],
    ):
        loading_values, values_list = self.load_back(
            all_hit_nodes,
        )
        return loading_values, values_list
        ""

    def match_prefix(self, params: MatchPrefixParams):
        all_hit_chunk_nodes = []
        host_hit_length = 0
        if params.key.is_kv_gen:
            for node in self.all_nodes:
                if node.is_preprocess_cache == params.key.is_preprocess_kv_gen:
                    prefix_text = params.key.prefix_prompt_text
                    text_without_prefix = params.key.origin_input_text[len(prefix_text):]
                    if text_without_prefix == node.text_without_prefix:
                        print(f"kv gen already run before\ntext={text_without_prefix}\n"
                              f"prefix={prefix_text}\n"
                              f"is_preprocess_cache={node.is_preprocess_cache}")
                        return MatchResult(
                            device_indices=torch.empty(
                                (0,),
                                dtype=torch.int64,
                                device=self.device,
                            ),  ## mengyao_debug let all be empty on the device.
                            all_hit_chunk_nodes=all_hit_chunk_nodes,
                            host_hit_length=host_hit_length,
                            last_host_node=None,
                            last_device_node=None,
                            no_need_to_run=True,
                        )

        if params.key.is_kv_gen and params.key.is_preprocess_kv_gen is False:
            # 如果是kv gen并且不是存储preprocess的话，不用匹配，直接生成
            input_text = ""
        else:
            # 如果是decode或者是preprocess的话，匹配前缀
            input_text = str(copy.deepcopy(params.key.prefix_prompt_text))
        last_round_found = True
        if len(input_text) == 0:
            print(f"mengyao_debug fusionrag cache match_prefix skipping prefix.")
        ##
        while len(input_text) > 0 and last_round_found:
            last_round_found = False
            for node in self.all_nodes:
                ## 找到和preprocess/raw 匹配的nodes
                if node.is_preprocess_cache == params.key.use_preprocess_kv_cache:
                    if len(node.text_without_prefix) > 20 and input_text.startswith(node.text_without_prefix):
                        print(f"load text: {node.text_without_prefix[:20]}, preprocess={node.is_preprocess_cache}, save_kv_cache={params.key.is_kv_gen}")
                        host_hit_length += len(node.host_value)
                        all_hit_chunk_nodes.append(node)
                        input_text = input_text[len(node.text_without_prefix) :]
                        last_round_found = True

        return MatchResult(
            device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ), ## mengyao_debug let all be empty on the device.
            all_hit_chunk_nodes=all_hit_chunk_nodes,
            host_hit_length=host_hit_length,
            last_host_node=None,
            last_device_node=None,
            no_need_to_run=False,
        )

    def insert(
        self,
        key: RadixKey,
        value=None,
        chunked: bool = False,
        priority: int | None = None,
        is_kv_gen: bool = False,
        kv_gen_prefix_len: int = 0,
        is_preprocess_cache: bool = False,
    ):
        if is_kv_gen:
            node = ChunkNode()
            ## 不存储prefix部分
            node.text_without_prefix = key.origin_input_text[len(key.prefix_prompt_text):]
            node.prefix_text = key.prefix_prompt_text
            node.values = [value]
            node.priority = priority
            node.cache_prefix_token_len = kv_gen_prefix_len
            node.is_preprocess_cache = is_preprocess_cache
            target_len = len(node.text_without_prefix)
            pos = bisect.bisect_left(
                self.all_nodes,
                -target_len,
                key=lambda n: -len(n.text_without_prefix)
            )
            self.all_nodes.insert(pos, node)
            ## 把数据写回主存里
            self.write_backup(node)
            ## 不留显存
            node.values = []


    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: Any,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        ""

    def terminate_prefetch(self, req_id: str):
        ""

    def check_prefetch_progress(self, req_id: str) -> bool:
        ""

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        ""

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def cache_unfinished_req(self, req: Req, chunked=False):
        print(f"cache_unfinished_req")
        ""

    ## fixme： 对于kvcache，在这里保存到ssd，并且保存到treecache里面；对于非kvcache，evict树；
    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        ## todo: 需要验证一下如果带了生成（max_token!=0）的话，要存哪些 kv_indices 是什么
        logger.error(
            f"fusionrag cache_finished_req: rid={req.rid} "
            f"is_kv_gen={req.is_kv_gen} save_raw_cache={req.save_raw_cache} "
            f"save_preprocess_cache={req.save_preprocess_cache} no_need_to_run={req.no_need_to_run}"
        )
        if req.no_need_to_run:
            return
        if req.is_kv_gen:
            ##todo：检查本地是否存在
            token_ids = req.origin_input_ids
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            radix_key = RadixKey(
                token_ids=token_ids,
                origin_input_text=req.origin_input_text,
                prefix_prompt_text=req.prefix_prompt,
            )
            values = kv_indices.to(dtype=torch.int64, copy=True)
            self.insert(
                radix_key,
                values[req.kv_gen_prefix_len:], ## 不存储prefix部分
                priority=0,
                is_kv_gen=True,
                kv_gen_prefix_len=req.kv_gen_prefix_len,
                is_preprocess_cache=req.save_preprocess_cache
            )
            self._write_cache_to_disk(req, kv_indices) ## 不存储prefix部分

        ## either case 都要把显存清理掉，要把output_ids部分也清理掉
        kv_committed_len = req.pop_committed_kv_cache()
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        ##todo 这里有问题，prefix_len 设置成了0，所以只把alloc_extend申请的内存free掉了，但是prefix的内存没有free掉。
        ##todo 还是把prefix_len改成对的吧
        self.cache_controller.mem_pool_device_allocator.free(kv_indices)
        for node_idx, node in enumerate(req.hit_chunk_nodes):
            value_to_remove = req.hit_chunk_values[node_idx].value
            try:
                for value_idx, value in enumerate(node.values):
                    if torch.equal(value, value_to_remove):  # 比较内容是否完全一致
                        del node.values[value_idx]
                        break
            except Exception as E:
                print(f"cache finish req error. E={E}, node={node}, value ={req.hit_chunk_values[i].value}")

        self.req_to_token_pool.free(req.req_pool_idx)

    def _write_cache_to_disk(self, req: Req, kv_indices_: torch.Tensor) -> None:
        kv_cache = []
        kv_indices = kv_indices_[req.kv_gen_prefix_len:] ## 不存储prefix部分
        for layer_id in range(self.kv_cache.layer_num):
            k_buffer = self.kv_cache.get_key_buffer(layer_id)[kv_indices].to('cpu')
            kv_cache.append(k_buffer)
        kv_cache = torch.stack(kv_cache, dim=0)
        text = req.origin_input_text
        prefix_prompt = req.prefix_prompt
        cache_prefix_token_len = req.kv_gen_prefix_len
        metadata = {
            "text": text[len(prefix_prompt):],  ## only save the document itself.
            "cache_prefix_token_len": cache_prefix_token_len,
            "prefix_text": prefix_prompt
        }
        ## 同步执行环境，不存在锁的问题
        md5_hash = hashlib.md5(text[len(prefix_prompt):].encode('utf-8')).hexdigest()
        if req.save_preprocess_cache is True:
            passage_kv_path = f"{self.preprocess_cache_path}/{md5_hash}"
            logger.error(
                "save to PREPROCESS cache\n"
                f"text=\n{text[len(prefix_prompt):]}\n"
                f"prefix=\n{prefix_prompt}"
            )
        elif req.save_raw_cache is True:
            passage_kv_path = f"{self.cache_path}/{md5_hash}"
            logger.error(
                "save to RAW cache\n"
                f"text=\n{text[len(prefix_prompt):][:20]}\n"
                f"prefix=\n{prefix_prompt}"
            )
        else:
            raise ValueError("either save_preprocess_cache or save_raw_cache must be True")
        logger.error(f"cache save path: {passage_kv_path}")
        os.makedirs(passage_kv_path, exist_ok=True)
        metadata_file_path = f"{passage_kv_path}/metadata.json"
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f)
        torch.save(kv_cache, f'{passage_kv_path}/{md5_hash}.pt')

    def dec_lock_ref(self, node: ChunkNode):
        ""

    def inc_lock_ref(self, node: ChunkNode):
        ""
