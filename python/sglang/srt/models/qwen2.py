# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import copy
import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

Qwen2Config = None


logger = logging.getLogger(__name__)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
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

        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
        output, _ = self.o_proj(attn_output)
        return output


    # def forward(
    #     self,
    #     positions: torch.Tensor,
    #     hidden_states: torch.Tensor,
    #     forward_batch: ForwardBatch,
    # ) -> torch.Tensor:
    #     qkv, _ = self.qkv_proj(hidden_states)
    #     q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    #     q, k = self.rotary_emb(positions, q, k)
    #     attn_output = self.attn(q, k, v, forward_batch)
    #     output, _ = self.o_proj(attn_output)
    #     return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
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
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                params_dtype=(
                    torch.float32
                    if get_global_server_args().rl_on_policy_target is not None
                    else None
                ),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
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
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
            )
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

        tp_size_ = get_tensor_model_parallel_world_size()
        cache_path_root = "/mnt/data3"
        # if not os.path.exists(cache_path_root):
        #     cache_path_root = "/mnt/data"
        model_name = getattr(config, "model_type", "qwen2")
        model_name = model_name.replace("/", "_")
        self.cache_path = f"/mnt/data3/shm/fusionrag/{model_name}/raw_kv_cache"
        self.preprocess_cache_path = (
            f"/mnt/data3/shm/fusionrag/{model_name}/preprocess_kv_cache"
        )
        if os.environ.get("DEBUG", "0") != "0":
            self.cache_path = (
                f"/mnt/data3/shm/fusionrag/{model_name}_tp_{tp_size_}/raw_kv_cache"
            )
            self.preprocess_cache_path = (
                f"/mnt/data3/shm/fusionrag/{model_name}_tp_{tp_size_}/preprocess_kv_cache"
            )

        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.preprocess_cache_path, exist_ok=True)

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        else:
            return self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        # if self.check_if_cache_calculated_before(forward_batch):
        #     return hidden_states

        # origin_positions = None
        # new_positions = self.load_kv_cache_to_hbm(
        #     forward_batch=forward_batch,
        #     dtype=hidden_states.dtype,
        #     device=hidden_states.device,
        #     positions=positions,
        # )
        # if new_positions is not None:
        #     origin_positions = copy.deepcopy(positions)
        #     positions = new_positions.to(positions.device)
        #     hidden_states = hidden_states[positions]

        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        # self.save_kv_cache_to_disk(
        #     forward_batch,
        #     dtype=hidden_states.dtype,
        #     positions=positions,
        # )

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )

    def check_if_cache_calculated_before(self, forward_batch):
        if forward_batch.reqs is not None and len(forward_batch.reqs) == 1:  ## only 1 task
            if (
                forward_batch.forward_mode == ForwardMode.EXTEND
                and forward_batch.reqs[0].fusionrag_params is not None
                and forward_batch.reqs[0].fusionrag_params.get("save_cache", False)
                is True
            ):  ## only happens when compute and save cache.
                text = forward_batch.reqs[0].origin_input_text
                prefix_prompt = forward_batch.reqs[0].fusionrag_params.get(
                    "prefix_prompt", ""
                )
                if (
                    forward_batch.reqs[0].fusionrag_params.get(
                        "save_preprocess_cache", False
                    )
                    is True
                ):
                    raw_cache = False
                else:
                    raw_cache = True
                all_chunk_cache = self.list_all_chunk_caches(raw_cache=raw_cache)
                all_texts = [cache[0] for cache in all_chunk_cache]
                if text[len(prefix_prompt) :] in all_texts:
                    if raw_cache:
                        print(
                            "[check_if_cache_calculated_before] RAW cache: text=\n"
                            f"{text[len(prefix_prompt):]}\n"
                            f"prefix=\n{prefix_prompt}\n"
                        )
                    else:
                        print(
                            "[check_if_cache_calculated_before] PREPROCESS cache: text=\n"
                            f"{text[len(prefix_prompt):]}\n"
                            f"prefix=\n{prefix_prompt}\n"
                        )
                    return True
        return False

    def save_kv_cache_to_disk(
        self,
        forward_batch,
        dtype,
        positions,
    ):
        _ = get_attention_tp_rank()
        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
        ):  ## only do this in prefill mode
            if forward_batch.extend_seq_lens_cpu is not None:
                for i, text_len in enumerate(forward_batch.extend_seq_lens_cpu):
                    k_cache = []
                    v_cache = []
                    text = forward_batch.reqs[i].origin_input_text
                    if (
                        forward_batch.reqs[i].fusionrag_params is not None
                        and forward_batch.reqs[i].fusionrag_params.get(
                            "save_cache", False
                        )
                        is True
                    ):
                        cache_start_idx = sum(forward_batch.extend_seq_lens_cpu[:i])
                        cache_end_idx = sum(
                            forward_batch.extend_seq_lens_cpu[: i + 1]
                        )
                        cache_prefix_token_len = len(
                            forward_batch.reqs[i].fusionrag_params["prefix_prompt_ids"]
                        )
                        prefix_prompt = forward_batch.reqs[i].fusionrag_params.get(
                            "prefix_prompt", ""
                        )
                        locs = forward_batch.out_cache_loc[
                            cache_start_idx + cache_prefix_token_len : cache_end_idx
                        ]
                        for layer_id in range(len(self.layers)):
                            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(
                                layer_id
                            ).to(dtype)[locs, :, :].to("cpu")
                            v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(
                                layer_id
                            ).to(dtype)[locs, :, :].to("cpu")
                            k_cache.append(k_buffer)
                            v_cache.append(v_buffer)
                        kv_cache = {
                            "k": torch.stack(k_cache, dim=0),
                            "v": torch.stack(v_cache, dim=0),
                        }
                        if not text.startswith(prefix_prompt):
                            print(
                                "mengyao_debug not start with prefix!\n"
                                f"prefix={forward_batch.reqs[i].fusionrag_params['prefix_prompt']}\n"
                                f"text={text}"
                            )
                        metadata = {
                            "text": text[len(prefix_prompt) :],  ## only save the document itself.
                            "cache_prefix_token_len": cache_prefix_token_len,
                            "prefix_text": prefix_prompt,
                        }
                        md5_hash = hashlib.md5(
                            text[len(prefix_prompt) :].encode("utf-8")
                        ).hexdigest()
                        if (
                            forward_batch.reqs[i].fusionrag_params.get(
                                "save_preprocess_cache", False
                            )
                            is True
                        ):
                            passage_kv_path = (
                                f"{self.preprocess_cache_path}/{md5_hash}"
                            )
                            print(
                                "mengyao_debug save to PREPROCESS cache\n"
                                f"text=\n{text[len(prefix_prompt):]}\n"
                                f"prefix=\n{prefix_prompt}"
                            )
                        else:
                            passage_kv_path = f"{self.cache_path}/{md5_hash}"
                            print(
                                "mengyao_debug save to RAW cache\n"
                                f"text=\n{text[len(prefix_prompt):]}\n"
                                f"prefix=\n{prefix_prompt}"
                            )
                        os.makedirs(passage_kv_path, exist_ok=True)
                        metadata_file_path = f"{passage_kv_path}/metadata.json"
                        with open(metadata_file_path, "w") as f:
                            json.dump(metadata, f)
                        torch.save(kv_cache, f"{passage_kv_path}/{md5_hash}.pt")

    def load_kv_cache_to_hbm(
        self,
        forward_batch,
        device,
        dtype,
        positions,
    ) -> torch.Tensor:
        tp_rank = get_attention_tp_rank()

        kv_cache = []
        ## starts from the easy case
        if (
            forward_batch.reqs is not None and len(forward_batch.reqs) == 1
        ):  ## only 1 task
            if (
                forward_batch.forward_mode == ForwardMode.EXTEND
            ):  ## this is a generation task, and it's in prefill mode
                input_text = copy.deepcopy(forward_batch.reqs[0].origin_input_text)
                load_raw_cache = True
                if (
                    forward_batch.reqs[0].fusionrag_params is not None
                    and forward_batch.reqs[0].fusionrag_params.get(
                        "load_preprocess_cache", False
                    )
                    is True
                ):
                    ## load from the preprocess cache
                    all_chunk_cache = self.list_all_chunk_caches(raw_cache=False)
                    load_raw_cache = False
                elif (
                    forward_batch.reqs[0].fusionrag_params is not None
                    and forward_batch.reqs[0].fusionrag_params.get(
                        "load_raw_cache", False
                    )
                    is True
                ):
                    ## load from raw kv cache
                    all_chunk_cache = self.list_all_chunk_caches(raw_cache=True)
                else:
                    return None
                found_prefix_chunk = True
                out_cache_loc_start_idx = 0
                out_cache_loc_end_idx = 0
                prefix_prompt = forward_batch.reqs[0].fusionrag_params.get(
                    "prefix_prompt", ""
                )
                recompute_idx = forward_batch.reqs[0].fusionrag_params.get(
                    "recompute_idx", []
                )
                prefix_prompt_len = len(prefix_prompt)
                ## 只允许load prefix_prompt_len 长度的prefix
                while found_prefix_chunk and prefix_prompt_len > 0:
                    found_prefix_chunk = False
                    for cache in all_chunk_cache:
                        chunk_text = cache[0]
                        kv_cache_pt = cache[1]
                        cache_prefix_token_len = cache[2]
                        if input_text.startswith(chunk_text):
                            found_prefix_chunk = True
                            chunk_tensor = torch.load(kv_cache_pt, weights_only=True)
                            if isinstance(chunk_tensor, dict):
                                k_chunk = chunk_tensor["k"].to(device)
                                v_chunk = chunk_tensor["v"].to(device)
                            elif (
                                isinstance(chunk_tensor, (list, tuple))
                                and len(chunk_tensor) == 2
                            ):
                                k_chunk = chunk_tensor[0].to(device)
                                v_chunk = chunk_tensor[1].to(device)
                            else:
                                raise ValueError(
                                    "Unexpected kv cache format. Expect dict with "
                                    "'k'/'v' or a (k, v) tuple."
                                )
                            out_cache_loc_end_idx += k_chunk.shape[1]
                            ## fix rope
                            load_tensor_len = k_chunk.shape[1]
                            prev_pos = torch.arange(
                                cache_prefix_token_len,
                                cache_prefix_token_len + load_tensor_len,
                                device=device,
                            )
                            cur_pos = torch.arange(
                                out_cache_loc_start_idx,
                                out_cache_loc_end_idx,
                                device=device,
                            )
                            print(f"doing rope from {prev_pos} to {cur_pos}")
                            for i, layer in enumerate(self.layers):
                                k = k_chunk[i]
                                v = v_chunk[i]
                                if (
                                    forward_batch.reqs[0].fusionrag_params.get(
                                        "rope", False
                                    )
                                    is True
                                ):
                                    k = correct_rope_rotation(
                                        k,
                                        layer.self_attn.rotary_emb.cos_sin_cache,
                                        wrong_positions=prev_pos,
                                        correct_positions=cur_pos,
                                    )
                                forward_batch.token_to_kv_pool.set_kv_buffer(
                                    layer.self_attn.attn,
                                    forward_batch.out_cache_loc[
                                        out_cache_loc_start_idx:out_cache_loc_end_idx
                                    ],
                                    k,
                                    v,
                                )
                            if load_raw_cache:
                                print(f"loaded one RAW cache, text=\n{chunk_text}")
                            else:
                                print(
                                    f"loaded one PREPROCESS cache, text=\n{chunk_text}"
                                )
                            out_cache_loc_start_idx = out_cache_loc_end_idx
                            input_text = input_text[len(chunk_text) :]
                            prefix_prompt_len -= len(chunk_text)
                            break
                if forward_batch.reqs[0].sampling_params.max_new_tokens > 0:
                    print(
                        "load_kv_cache_to_hbm start computing from "
                        f"{out_cache_loc_end_idx}"
                    )
                if len(input_text) == len(forward_batch.reqs[0].origin_input_text):
                    ## match nothing.
                    return None
                ##fixme: only support one input.
                if out_cache_loc_end_idx == len(
                    forward_batch.reqs[0].origin_input_ids
                ):  ## compute at least one.
                    out_cache_loc_end_idx -= 1
                sorted_indices = torch.arange(
                    out_cache_loc_end_idx,
                    len(forward_batch.reqs[0].origin_input_ids),
                )
                if len(recompute_idx) > 0:
                    recompute_idx_tensor = torch.tensor(recompute_idx)
                    sorted_indices = torch.cat(
                        (sorted_indices, recompute_idx_tensor)
                    )
                    sorted_indices = torch.sort(sorted_indices).values
                forward_batch.fusion_rag_indices = sorted_indices
                print(f"recompute_idx = {forward_batch.fusion_rag_indices}")
                print(
                    f"recompute percentage = {len(recompute_idx)/out_cache_loc_end_idx}"
                )
                return sorted_indices
        return None

    def list_all_chunk_caches(self, raw_cache: bool):
        result = []
        if raw_cache:
            cache_path = self.cache_path
        else:
            cache_path = self.preprocess_cache_path
        for folder in os.listdir(cache_path):
            folder_path = os.path.join(cache_path, folder)  ## folder is the md5
            if os.path.isdir(folder_path):
                metadata_path = os.path.join(folder_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        text = metadata.get("text", "")
                        cache_prefix_token_len = metadata.get(
                            "cache_prefix_token_len", ""
                        )
                        result.append(
                            (text, os.path.join(folder_path, f"{folder}.pt"), cache_prefix_token_len)
                        )

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error reading {metadata_path}: {e}")

        return result


class Qwen2ForCausalLM(nn.Module):
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
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2Model(
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
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

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
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
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


EntryClass = Qwen2ForCausalLM


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
    cos_wrong = wrong_cache[:, :half_dim].unsqueeze(1)
    sin_wrong = wrong_cache[:, half_dim:].unsqueeze(1)

    # 2. 获取正确位置的cos/sin
    correct_cache = rotary_cache[correct_positions]
    cos_correct = correct_cache[:, :half_dim].unsqueeze(1)
    sin_correct = correct_cache[:, half_dim:].unsqueeze(1)

    # 3. 计算旋转差值的cos/sin
    cos_delta = cos_correct * cos_wrong + sin_correct * sin_wrong
    sin_delta = sin_correct * cos_wrong - cos_correct * sin_wrong

    # 4. 将k_wrong重塑为[..., half_dim, 2]
    k_reshaped = k_wrong.view(seq_len, num_heads_k, half_dim, 2)

    # 5. 分离偶数和奇数维度
    k_even = k_reshaped[..., 0]
    k_odd = k_reshaped[..., 1]

    # 6. 应用修正旋转
    k_correct_even = k_even * cos_delta - k_odd * sin_delta
    k_correct_odd = k_even * sin_delta + k_odd * cos_delta

    # 7. 重新组合
    k_correct = torch.stack([k_correct_even, k_correct_odd], dim=-1)

    return k_correct.view(seq_len, num_heads_k, head_dim).to(k_wrong.dtype)
