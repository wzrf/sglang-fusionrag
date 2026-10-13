# Adapted from qwen2.py
import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch
import numpy as np
import triton
import triton.language as tl
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors, ForwardMode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

    from sglang.srt.hardware_backend.npu.cmo import get_cmo_stream, wait_cmo_stream


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        # q_ = copy.deepcopy(q)
        # k_ = copy.deepcopy(k)
        # positions_0 = torch.zeros(len(positions), dtype=torch.int32)
        #
        # num_tokens = len(positions)
        # q_ = q_.view(num_tokens, -1, self.head_dim)
        # k_ = k_.view(num_tokens, -1, self.head_dim)
        # q_ = correct_rope_rotation(q_, self.rotary_emb.cos_sin_cache, positions_0, positions)
        # k_ = correct_rope_rotation(k_, self.rotary_emb.cos_sin_cache, positions_0, positions)
        # q_ = q_.view(num_tokens, -1)
        # k_ = k_.view(num_tokens, -1)

        q, k = self.rotary_emb(positions, q, k)
        #
        # diff = compare_relative_difference_torch(k, k_)
        # print(diff)

        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        if not _is_npu or forward_batch.forward_mode.is_extend():
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if get_global_server_args().rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        def has_duplicates(tensor):
            # 展平后比较元素总数与唯一元素个数
            return tensor.numel() != torch.unique(tensor).numel()

        save_kv_cache = True
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            save_kv_cache = False
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer=self.attn,
                loc=forward_batch.out_cache_loc,
                cache_k=k,
                cache_v=v,
            )
            kv_indices = forward_batch.fetch_mha_one_shot_kv_indices()
            k_, v_ = forward_batch.token_to_kv_pool.get_kv_buffer(self.attn.layer_id)
            k_ = k_[kv_indices]
            v_ = v_[kv_indices]
            k = k_.flatten(start_dim=-2).contiguous()
            v = v_.flatten(start_dim=-2).contiguous()
            if self.attn.layer_id == 0:
                print(f"EXTEND k size = {k.size()}, q.size() = {q.size()}")
                if has_duplicates(kv_indices):
                    print(f"mengyao_debug kv_indices HAS DUPLICATES, kv_indices={kv_indices}")


        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            post_residual_addition=post_residual_addition,
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            cache=(
                [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                if _is_npu
                and not get_global_server_args().enable_piecewise_cuda_graph
                and (
                    hasattr(self.mlp.gate_up_proj, "weight")
                    and hasattr(self.mlp.down_proj, "weight")
                )
                else None
            ),
        )
        hidden_states = self.mlp(hidden_states)
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )


class Qwen3ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # self.fix_rope_test(forward_batch)
        self.fix_rope_test_triton(forward_batch)
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


    def fix_rope_test(
        self,
        forward_batch
    ):
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return
        if forward_batch is None or forward_batch.reqs is None:
            return
        if forward_batch.rope_fixed:
            return
        for req in forward_batch.reqs:
            if req.is_kv_gen:
                continue
            if req.hit_chunk_values is not None:
                for layer_id, layer in enumerate(self.model.layers):
                    for hit_chunk_node in req.hit_chunk_values:
                        # ## fixme: mengyao_debug 这里可能有bug
                        # if hit_chunk_node.original_position[0] > hit_chunk_node.current_position[0]:
                        device_indices = hit_chunk_node.value
                        k_, v_ = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)
                        k = k_[device_indices]
                        v = v_[device_indices]
                        k_rope = correct_rope_rotation(k,
                                                       layer.self_attn.rotary_emb.cos_sin_cache,
                                                       wrong_positions=hit_chunk_node.original_position,
                                                       correct_positions=hit_chunk_node.current_position)
                        forward_batch.token_to_kv_pool.set_kv_buffer(
                            layer=layer.self_attn.attn,
                            loc=device_indices,
                            cache_k=k_rope,
                            cache_v=v,
                        )
        forward_batch.rope_fixed = True

    @torch.no_grad()
    def fix_rope_test_triton(self, forward_batch):
        if forward_batch is None or forward_batch.reqs is None:
            return

        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return

        if forward_batch.rope_fixed:
            return

        all_device_indices = []
        all_wrong_positions = []
        all_correct_positions = []

        # 1. 收集所有 Request 的 Chunk
        for req in forward_batch.reqs:
            if req.is_kv_gen:
                continue

            if not req.hit_chunk_values:
                continue

            for hit_chunk_node in req.hit_chunk_values:
                all_device_indices.append(hit_chunk_node.value)
                all_wrong_positions.append(hit_chunk_node.original_position)
                all_correct_positions.append(hit_chunk_node.current_position)

        if not all_device_indices:
            forward_batch.rope_fixed = True
            return

        # 2. 使用 KV Cache 所在的 GPU
        first_kv, _ = forward_batch.token_to_kv_pool.get_kv_buffer(
            self.model.start_layer
        )
        device = first_kv.device

        device_indices = torch.cat([
            torch.as_tensor(x, device=device, dtype=torch.int64).reshape(-1)
            for x in all_device_indices
        ]).contiguous()

        wrong_positions = torch.cat([
            torch.as_tensor(x, device=device, dtype=torch.int32).reshape(-1)
            for x in all_wrong_positions
        ]).contiguous()

        correct_positions = torch.cat([
            torch.as_tensor(x, device=device, dtype=torch.int32).reshape(-1)
            for x in all_correct_positions
        ]).contiguous()

        assert (
            device_indices.numel()
            == wrong_positions.numel()
            == correct_positions.numel()
        )

        # 调试阶段检查重复 KV 槽位
        assert (
            torch.unique(device_indices).numel()
            == device_indices.numel()
        ), "Duplicate KV cache indices detected!"

        # 3. 每个 Layer 调用一次 Triton Kernel
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]

            k_buffer, _ = (
                forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)
            )

            rotary_cache = layer.self_attn.rotary_emb.cos_sin_cache

            rope_dim = layer.self_attn.head_dim

            assert k_buffer.shape[-1] == rope_dim

            correct_rope_chunks_neox_triton(
                kv_buffer=k_buffer,
                rotary_cache=rotary_cache,
                device_indices=device_indices,
                wrong_positions=wrong_positions,
                correct_positions=correct_positions,
                rope_dim=rope_dim,
            )

        # 4. 所有 Layer 修正完成
        forward_batch.rope_fixed = True

        print(
            f"[Qwen3 fix_rope] "
            f"chunks={len(all_device_indices)}, "
            f"tokens={device_indices.numel()}, "
            f"layers={self.model.end_layer - self.model.start_layer}"
        )


def correct_rope_rotation(k_wrong, rotary_cache, wrong_positions, correct_positions):
    """
    修正错误的RoPE旋转
    k_wrong: 已经用wrong_positions旋转过的张量 [seq_len, num_heads, head_dim]
    rotary_cache: [max_position, head_dim]，前一半是cos，后一半是sin
    wrong_positions: 之前错误使用的位置
    correct_positions: 正确的位置
    """
    seq_len, num_heads_k, head_dim = k_wrong.shape
    half_dim = head_dim // 2

    # 1. 获取错误位置的cos/sin
    wrong_cache = rotary_cache[wrong_positions]
    cos_wrong = wrong_cache[:, :half_dim].unsqueeze(1).to(k_wrong.dtype)
    sin_wrong = wrong_cache[:, half_dim:].unsqueeze(1).to(k_wrong.dtype)

    # 2. 获取正确位置的cos/sin
    correct_cache = rotary_cache[correct_positions]
    cos_correct = correct_cache[:, :half_dim].unsqueeze(1)
    sin_correct = correct_cache[:, half_dim:].unsqueeze(1)

    # 3. 计算旋转差值的cos/sin
    cos_delta = cos_correct * cos_wrong + sin_correct * sin_wrong
    sin_delta = sin_correct * cos_wrong - cos_correct * sin_wrong

    cos_delta = cos_delta.to(k_wrong.dtype)
    sin_delta = sin_delta.to(k_wrong.dtype)

    ##fixme:  这个对应的是neox
    k_even, k_odd = torch.chunk(k_wrong, 2, dim=-1)

    # 6. 应用修正旋转
    k_correct_even = k_even * cos_delta - k_odd * sin_delta
    k_correct_odd = k_even * sin_delta + k_odd * cos_delta

    # 7. 重新组合
    k_correct = torch.cat((k_correct_even, k_correct_odd), dim=-1)

    return k_correct.view(seq_len, num_heads_k, head_dim).to(k_wrong.dtype).contiguous()


def compare_relative_difference_torch(a, b, mode='b', eps=1e-8):
    a = a.detach().float()
    b = b.detach().float()
    diff_abs = torch.abs(a - b)

    if mode == 'b':
        denom = torch.abs(b) + eps
    elif mode == 'max':
        denom = torch.max(torch.abs(a), torch.abs(b)) + eps
    elif mode == 'mean':
        denom = (torch.abs(a) + torch.abs(b)) / 2 + eps
    else:
        raise ValueError("mode must be 'b', 'max', or 'mean'")

    rel_diff = diff_abs / denom

    max_val = torch.max(rel_diff)
    idx_flat = torch.argmax(rel_diff)
    position = np.unravel_index(idx_flat.cpu().numpy(), a.shape)

    mean_val = torch.mean(rel_diff)

    return {
        'max_rel_diff': max_val.item(),
        'max_position': position,
        'max_value_a': a.flatten()[idx_flat].item(),
        'max_value_b': b.flatten()[idx_flat].item(),
        'mean_rel_diff': mean_val.item()
    }


@triton.jit
def _correct_rope_chunks_neox_kernel(
    KV,
    RotaryCache,
    DeviceIndices,
    WrongPositions,
    CorrectPositions,
    H: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    KV_STRIDE_0: tl.constexpr,
    KV_STRIDE_1: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 一个 Triton Program 处理一个 token
    token_id = tl.program_id(0)

    half_dim: tl.constexpr = ROPE_DIM // 2

    # KV Cache 中的真实槽位
    device_idx = tl.load(DeviceIndices + token_id)

    # 原始位置和目标位置
    wrong_pos = tl.load(WrongPositions + token_id)
    correct_pos = tl.load(CorrectPositions + token_id)

    # 并行处理所有 KV heads 和 rotary pairs
    idx = tl.arange(0, BLOCK)
    mask = idx < H * half_dim

    head_id = idx // half_dim
    pair_id = idx % half_dim

    # 读取原始位置的 cos/sin
    wrong_base = RotaryCache + wrong_pos * CACHE_STRIDE

    cw = tl.load(
        wrong_base + pair_id,
        mask,
        other=1,
    ).to(tl.float32)

    sw = tl.load(
        wrong_base + half_dim + pair_id,
        mask,
        other=0,
    ).to(tl.float32)

    # 读取目标位置的 cos/sin
    correct_base = RotaryCache + correct_pos * CACHE_STRIDE

    cc = tl.load(
        correct_base + pair_id,
        mask,
        other=1,
    ).to(tl.float32)

    sc = tl.load(
        correct_base + half_dim + pair_id,
        mask,
        other=0,
    ).to(tl.float32)

    # 计算相对旋转角度
    cos_delta = cc * cw + sc * sw
    sin_delta = sc * cw - cc * sw

    # NeoX RoPE:
    # 前半维度与后半维度配对
    #
    # [0, 1, ..., half_dim-1]
    # [half_dim, ..., ROPE_DIM-1]

    kv_offset = (
        device_idx * KV_STRIDE_0
        + head_id * KV_STRIDE_1
        + pair_id
    )

    k_even = tl.load(
        KV + kv_offset,
        mask,
        other=0,
    ).to(tl.float32)

    k_odd = tl.load(
        KV + kv_offset + half_dim,
        mask,
        other=0,
    ).to(tl.float32)

    # 应用旋转修正
    out_even = (
        k_even * cos_delta
        - k_odd * sin_delta
    )

    out_odd = (
        k_even * sin_delta
        + k_odd * cos_delta
    )

    # 原地写回 K Cache
    tl.store(
        KV + kv_offset,
        out_even,
        mask,
    )

    tl.store(
        KV + kv_offset + half_dim,
        out_odd,
        mask,
    )


def correct_rope_chunks_neox_triton(
    kv_buffer,
    rotary_cache,
    device_indices,
    wrong_positions,
    correct_positions,
    rope_dim=None,
):
    """
    合并多个 Chunk，使用一次 Triton Kernel
    原地修正 Qwen3 K Cache 的 NeoX RoPE。

    Args:
        kv_buffer:
            [pool_size, num_kv_heads, head_dim]
            原始 KV Cache 中的 K Tensor。

        rotary_cache:
            [max_position, rope_dim]
            前半部分为 cos，后半部分为 sin。

        device_indices:
            [total_tokens]
            合并后的 KV Cache 槽位索引。

        wrong_positions:
            [total_tokens]
            K 原来使用的 RoPE position。

        correct_positions:
            [total_tokens]
            K 应当使用的 RoPE position。

        rope_dim:
            RoPE 旋转维度，默认等于 head_dim。

    Returns:
        kv_buffer:
            原地修改后的 K Cache。
    """

    assert kv_buffer.is_cuda
    assert kv_buffer.ndim == 3
    assert kv_buffer.stride(-1) == 1

    assert rotary_cache.is_cuda
    assert rotary_cache.is_contiguous()

    assert device_indices.is_cuda
    assert wrong_positions.is_cuda
    assert correct_positions.is_cuda

    assert device_indices.is_contiguous()
    assert wrong_positions.is_contiguous()
    assert correct_positions.is_contiguous()

    pool_size, num_heads, head_dim = kv_buffer.shape

    if rope_dim is None:
        rope_dim = head_dim

    assert rope_dim == head_dim
    assert rope_dim % 2 == 0
    assert rotary_cache.shape[1] == rope_dim

    total_tokens = device_indices.numel()

    assert wrong_positions.numel() == total_tokens
    assert correct_positions.numel() == total_tokens

    if total_tokens == 0:
        return kv_buffer

    block = triton.next_power_of_2(
        num_heads * (rope_dim // 2)
    )

    _correct_rope_chunks_neox_kernel[(total_tokens,)](
        kv_buffer,
        rotary_cache,
        device_indices,
        wrong_positions,
        correct_positions,
        num_heads,
        rope_dim,
        kv_buffer.stride(0),
        kv_buffer.stride(1),
        rotary_cache.stride(0),
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )

    return kv_buffer


EntryClass = Qwen3ForCausalLM
