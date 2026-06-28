# Copyright 2025 The Gigaworld Team and The HuggingFace Team. All rights reserved.
#
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
#
# Additional contributions by zkey@GigaAI.

import glob
import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import apply_lora_scale, deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph

from .gigaworld_kernels import attn_varlen_func, create_navit_attention_masks

# ANSI color codes
_C = {
    "reset": "\033[0m",
    "dim": "\033[90m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
}

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)


def apply_rotary_emb_transposed(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(hidden_states)


def _get_qkv_projections(attn: "GigaworldAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if not attn.is_cross_attention:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value

class Discriminator3DHead(nn.Module):
    def __init__(self, input_channel, cond_map_dim=768):
        super().__init__()

        self.head3d = nn.Sequential(
            nn.Conv3d(input_channel, cond_map_dim, 3, stride=(1, 1, 1), padding=(1, 1, 1)),  # [31, 8, 8]
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(cond_map_dim, cond_map_dim, 4, stride=[2, 2, 2], padding=(1, 1, 1)),  #  [15, 4, 4]
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(cond_map_dim, cond_map_dim, 4, stride=[2, 2, 2], padding=(1, 1, 1)),  #  [7, 2, 2]
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(cond_map_dim, cond_map_dim, 3, stride=[2, 1, 1], padding=(1, 1, 1)),  #  [3, 2, 2]
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(cond_map_dim, cond_map_dim, 3, stride=[2, 1, 1], padding=(1, 1, 1)),  #  [1, 2, 2]
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(
                cond_map_dim, cond_map_dim, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=(0, 1, 1)
            ),  #  [b, 768, 1, 1, 2]
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(cond_map_dim, 1),
        )

    def forward(self, x):
        return self.head3d(x)

class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device="cuda",
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)

class GigaworldOutputNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, original_context_length: int):
        temb = temb[:, -original_context_length:, :]
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2).to(hidden_states.device), scale.squeeze(2).to(hidden_states.device)
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = (self.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        return hidden_states

class GigaworldAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "GigaworldAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

        self.kv_cache = None
        self.cache_enabled = False

    def enable_cache(self):
        self.cache_enabled = True
        self.kv_cache = None

    def disable_cache(self):
        self.cache_enabled = False
        self.kv_cache = None

    def clear_cache(self):
        self.kv_cache = None

    def __call__(
        self,
        attn: "GigaworldAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        original_context_length: int = None,
        original_context_length_list: list = None,
        enable_navit: bool = False,
        is_first_denoising_step: bool = False,
    ) -> torch.Tensor:
        use_cache = False
        history_seq_len = None
        enable_cross = attn.is_cross_attention

        if not enable_cross:
            history_seq_len = (hidden_states.shape[1] - original_context_length) // len(original_context_length_list)

        if attn.restrict_self_attn:
            use_cache = self.cache_enabled and not is_first_denoising_step and self.kv_cache is not None
            assert not (use_cache and enable_navit), "Cache and NAViT are incompatible"

            if use_cache:
                key_history = self.kv_cache["key_history"]
                value_history = self.kv_cache["value_history"]
                history_hidden_states = self.kv_cache["history_hidden_states"]

                hidden_states = hidden_states[:, history_seq_len:]
                rotary_emb = rotary_emb[:, history_seq_len:] if rotary_emb is not None else None

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.restrict_self_attn and not use_cache:
            if enable_navit:
                seq_start = 0
                num_seqs = len(original_context_length_list)
                query_list = [None] * num_seqs
                key_list = [None] * num_seqs
                value_list = [None] * num_seqs
                query_history_list = [None] * num_seqs
                key_history_list = [None] * num_seqs
                value_history_list = [None] * num_seqs

                if attn.restrict_lora:
                    history_hidden_states_list = [None] * num_seqs

                if rotary_emb is not None:
                    rotary_emb_list = [None] * num_seqs
                    history_rotary_emb_list = [None] * num_seqs

                for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                    seq_end = seq_start + cur_seq_len + history_seq_len

                    slice_qkv = slice(seq_start, seq_end)
                    cur_query = query[:, slice_qkv, :]
                    cur_key = key[:, slice_qkv, :]
                    cur_value = value[:, slice_qkv, :]

                    query_history_list[idx] = cur_query[:, :history_seq_len]
                    query_list[idx] = cur_query[:, history_seq_len:]

                    key_history_list[idx] = cur_key[:, :history_seq_len]
                    key_list[idx] = cur_key[:, history_seq_len:]

                    value_history_list[idx] = cur_value[:, :history_seq_len]
                    value_list[idx] = cur_value[:, history_seq_len:]

                    if attn.restrict_lora:
                        cur_hidden = hidden_states[:, slice_qkv, :]
                        history_hidden_states_list[idx] = cur_hidden[:, :history_seq_len]

                    if rotary_emb is not None:
                        cur_rotary_emb = rotary_emb[:, slice_qkv, :]
                        history_rotary_emb_list[idx] = cur_rotary_emb[:, :history_seq_len]
                        rotary_emb_list[idx] = cur_rotary_emb[:, history_seq_len:]

                    seq_start = seq_end

                query = torch.cat(query_list, dim=1)
                key = torch.cat(key_list, dim=1)
                value = torch.cat(value_list, dim=1)
                query_history = torch.cat(query_history_list, dim=1)
                key_history = torch.cat(key_history_list, dim=1)
                value_history = torch.cat(value_history_list, dim=1)

                if attn.restrict_lora:
                    history_hidden_states = torch.cat(history_hidden_states_list, dim=1)
                    query_history = query_history + attn.q_loras(history_hidden_states)
                    key_history = key_history + attn.k_loras(history_hidden_states)
                    value_history = value_history + attn.v_loras(history_hidden_states)

                query_history = query_history.unflatten(2, (attn.heads, -1))
                key_history = key_history.unflatten(2, (attn.heads, -1))
                value_history = value_history.unflatten(2, (attn.heads, -1))

                if rotary_emb is not None:
                    rotary_emb = torch.cat(rotary_emb_list, dim=1)
                    history_rotary_emb = torch.cat(history_rotary_emb_list, dim=1)
                    query_history = apply_rotary_emb_transposed(query_history, history_rotary_emb)
                    key_history = apply_rotary_emb_transposed(key_history, history_rotary_emb)
            else:
                history_hidden_states = hidden_states[:, :history_seq_len]
                query_history, query = query[:, :history_seq_len], query[:, history_seq_len:]
                key_history, key = key[:, :history_seq_len], key[:, history_seq_len:]
                value_history, value = value[:, :history_seq_len], value[:, history_seq_len:]

                if attn.restrict_lora:
                    query_history = query_history + attn.q_loras(history_hidden_states)
                    key_history = key_history + attn.k_loras(history_hidden_states)
                    value_history = value_history + attn.v_loras(history_hidden_states)

                query_history = query_history.unflatten(2, (attn.heads, -1))
                key_history = key_history.unflatten(2, (attn.heads, -1))
                value_history = value_history.unflatten(2, (attn.heads, -1))

                if rotary_emb is not None:
                    history_rotary_emb, rotary_emb = (rotary_emb[:, :history_seq_len], rotary_emb[:, history_seq_len:])
                    query_history = apply_rotary_emb_transposed(query_history, history_rotary_emb)
                    key_history = apply_rotary_emb_transposed(key_history, history_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = apply_rotary_emb_transposed(query, rotary_emb)
            key = apply_rotary_emb_transposed(key, rotary_emb)

        if attn.restrict_self_attn:
            if use_cache:
                key = torch.cat([key_history, key], dim=1)
                value = torch.cat([value_history, value], dim=1)
            else:
                if enable_navit:
                    num_seqs = len(original_context_length_list)

                    key_list = [None] * num_seqs
                    value_list = [None] * num_seqs

                    seq_start = 0
                    seq_start_history = 0

                    for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                        key_list[idx] = torch.cat(
                            [
                                key_history[:, seq_start_history : seq_start_history + history_seq_len, :],
                                key[:, seq_start : seq_start + cur_seq_len, :],
                            ],
                            dim=1,
                        )

                        value_list[idx] = torch.cat(
                            [
                                value_history[:, seq_start_history : seq_start_history + history_seq_len, :],
                                value[:, seq_start : seq_start + cur_seq_len, :],
                            ],
                            dim=1,
                        )

                        seq_start += cur_seq_len
                        seq_start_history += history_seq_len

                    key = torch.cat(key_list, dim=1)
                    value = torch.cat(value_list, dim=1)
                    import ipdb; ipdb.set_trace()  # DEBUG
                    history_hidden_states = attn_varlen_func(
                        query_history,
                        key_history,
                        value_history,
                        attention_mask=attention_mask[1],
                    )
                else:
                    key = torch.cat([key_history, key], dim=1)
                    value = torch.cat([value_history, value], dim=1)

                    history_hidden_states = attn_varlen_func(
                        query_history,
                        key_history,
                        value_history,
                    )
                history_hidden_states = history_hidden_states.flatten(2, 3)
                history_hidden_states = history_hidden_states.type_as(query)

                if self.cache_enabled and is_first_denoising_step and not enable_navit:
                    self.kv_cache = {
                        "key_history": key_history,
                        "value_history": value_history,
                        "history_hidden_states": history_hidden_states,
                    }

        if enable_cross and enable_navit:
            key = key.repeat(1, len(original_context_length_list), 1, 1)
            value = value.repeat(1, len(original_context_length_list), 1, 1)

        if not enable_cross and history_seq_len > 0 and attn.is_amplify_history:
            scale_key = attn.get_scale_key()
            if attn.history_scale_mode == "per_head":
                scale_key = scale_key.view(1, 1, -1, 1)

            if enable_navit:
                key_new = key.clone()
                seq_start = 0
                for cur_seq_len in original_context_length_list[::-1]:
                    hist_slice = slice(seq_start, seq_start + history_seq_len)
                    key_new[:, hist_slice] = key[:, hist_slice] * scale_key
                    seq_start += history_seq_len + cur_seq_len
                key = key_new
            else:
                key = torch.cat([key[:, :history_seq_len] * scale_key, key[:, history_seq_len:]], dim=1)

        hidden_states = attn_varlen_func(
            query,
            key,
            value,
            attention_mask=attention_mask[0] if isinstance(attention_mask, list) else attention_mask,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if attn.restrict_self_attn:
            if enable_navit:
                num_seqs = len(original_context_length_list)
                hidden_states_list = [None] * num_seqs

                seq_start = 0
                seq_start_history = 0

                for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                    hidden_states_list[idx] = torch.cat(
                        [
                            history_hidden_states[:, seq_start_history : seq_start_history + history_seq_len, :],
                            hidden_states[:, seq_start : seq_start + cur_seq_len, :],
                        ],
                        dim=1,
                    )

                    seq_start += cur_seq_len
                    seq_start_history += history_seq_len

                hidden_states = torch.cat(hidden_states_list, dim=1)
            else:
                hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class GigaworldAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The GigaworldAttnProcessor2_0 class is deprecated and will be removed in a future version. "
            "Please use GigaworldAttnProcessor instead. "
        )
        deprecate("GigaworldAttnProcessor2_0", "1.0.0", deprecation_message, standard_warn=False)
        return GigaworldAttnProcessor(*args, **kwargs)

class GigaworldAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = GigaworldAttnProcessor
    _available_processors = [GigaworldAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
        restrict_self_attn=False,
        is_train_restrict_lora=False,
        restrict_lora=False,
        restrict_lora_rank=128,
        is_amplify_history=False,
        history_scale_mode="per_head",  # [scalar, per_head]
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

        self.restrict_self_attn = restrict_self_attn
        self.restrict_lora = restrict_lora
        if restrict_lora:
            self.init_lora(is_train=is_train_restrict_lora, lora_rank=restrict_lora_rank)

        self.is_amplify_history = is_amplify_history
        if is_amplify_history:
            if history_scale_mode == "scalar":
                self.history_key_scale = nn.Parameter(torch.ones(1))
            elif history_scale_mode == "per_head":
                self.history_key_scale = nn.Parameter(torch.ones(heads))
            else:
                raise ValueError(f"Unknown history_scale_mode: {history_scale_mode}")
            self.history_scale_mode = history_scale_mode
            self.max_scale = 10.0
            self.register_buffer("_scale_cache", None)

    def get_scale_key(self):
        if self.history_key_scale.requires_grad:
            scale = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_scale - 1.0)
        else:
            if self._scale_cache is None:
                self._scale_cache = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_scale - 1.0)
            scale = self._scale_cache
        return scale

    def init_lora(self, is_train=False, lora_rank=128):
        dim = self.inner_dim
        self.q_loras = LoRALinearLayer(dim, dim, rank=lora_rank)
        self.k_loras = LoRALinearLayer(dim, dim, rank=lora_rank)
        self.v_loras = LoRALinearLayer(dim, dim, rank=lora_rank)

        requires_grad = is_train
        for lora in [self.q_loras, self.k_loras, self.v_loras]:
            for param in lora.parameters():
                param.requires_grad = requires_grad

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if not self.is_cross_attention:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        original_context_length: int = None,
        original_context_length_list: list = None,
        enable_navit: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            original_context_length,
            original_context_length_list,
            enable_navit,
            **kwargs,
        )

class GigaworldTimeTextEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        is_return_encoder_hidden_states: bool = True,
    ):
        B = None
        F = None
        if timestep.ndim == 2:
            B, F = timestep.shape
            timestep = timestep.flatten()

        timestep = self.timesteps_proj(timestep)  # torch.Size([2]) -> torch.Size([2, 256])

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)  # torch.Size([2, 1536])
        timestep_proj = self.time_proj(self.act_fn(temb))  # torch.Size([2, 9216]

        if B is not None and F is not None:
            temb = temb.reshape(B, F, -1)
            timestep_proj = timestep_proj.reshape(B, F, -1)

        if encoder_hidden_states is not None and is_return_encoder_hidden_states:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)  # torch.Size([2, 512, 1536])

        return temb, timestep_proj, encoder_hidden_states

class GigaworldRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta
        self.register_buffer("freqs_base_t", self._get_freqs_base(self.DT), persistent=False)
        self.register_buffer("freqs_base_y", self._get_freqs_base(self.DY), persistent=False)
        self.register_buffer("freqs_base_x", self._get_freqs_base(self.DX), persistent=False)

    def _get_freqs_base(self, dim):
        return 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))

    @torch.no_grad()
    def get_frequency_batched(self, freqs_base, pos):
        freqs = torch.einsum("d,bthw->dbthw", freqs_base, pos)
        freqs = freqs.repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    @lru_cache(maxsize=32)
    def _get_spatial_meshgrid(self, height, width, device_str):
        device = torch.device(device_str)
        gy = torch.arange(height, device=device, dtype=torch.float32)
        gx = torch.arange(width, device=device, dtype=torch.float32)
        GY, GX = torch.meshgrid(gy, gx, indexing="ij")
        return GY, GX

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        B = frame_indices.shape[0]
        T = frame_indices.shape[1]

        frame_indices = frame_indices.to(device=device, dtype=torch.float32)
        GY, GX = self._get_spatial_meshgrid(height, width, str(device))

        GT = frame_indices[:, :, None, None].expand(B, T, height, width)
        GY_batch = GY[None, None, :, :].expand(B, T, -1, -1)
        GX_batch = GX[None, None, :, :].expand(B, T, -1, -1)

        FCT, FST = self.get_frequency_batched(self.freqs_base_t, GT)
        FCY, FSY = self.get_frequency_batched(self.freqs_base_y, GY_batch)
        FCX, FSX = self.get_frequency_batched(self.freqs_base_x, GX_batch)

        result = torch.cat([FCT, FCY, FCX, FST, FSY, FSX], dim=0)

        return result.permute(1, 0, 2, 3, 4)

@maybe_allow_in_graph
class GigaworldTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        restrict_self_attn: bool = False,
        guidance_cross_attn: bool = False,
        is_train_restrict_lora: bool = False,
        restrict_lora: bool = False,
        restrict_lora_rank: int = 128,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",  # [scalar, per_head],
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = GigaworldAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=GigaworldAttnProcessor(),
            restrict_self_attn=restrict_self_attn,
            is_train_restrict_lora=is_train_restrict_lora,
            restrict_lora=restrict_lora,
            restrict_lora_rank=restrict_lora_rank,
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
        )

        # 2. Cross-attention
        self.attn2 = GigaworldAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=GigaworldAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # 4. Guidance cross-attention
        self.guidance_cross_attn = guidance_cross_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        navit_hidden_attention_mask: Optional[torch.Tensor] = None,
        navit_encoder_attention_mask: Optional[torch.Tensor] = None,
        original_context_length: int = None,
        original_context_length_list: list = None,
        is_first_denoising_step: bool = False,
    ) -> torch.Tensor:
        enable_navit = False
        if len(original_context_length_list) > 1:
            enable_navit = True

        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            None,
            navit_hidden_attention_mask,
            rotary_emb,
            original_context_length,
            original_context_length_list,
            enable_navit,
            is_first_denoising_step=is_first_denoising_step,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        if self.guidance_cross_attn:
            history_seq_len = (hidden_states.shape[1] - original_context_length) // len(original_context_length_list)

            if enable_navit:
                num_seqs = len(original_context_length_list)

                hidden_states_list = [None] * num_seqs
                history_hidden_states_list = [None] * num_seqs

                seq_start = 0
                for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                    seq_end = seq_start + cur_seq_len + history_seq_len
                    cur_hidden_states = hidden_states[:, seq_start:seq_end, :]

                    history_hidden_states_list[idx] = cur_hidden_states[:, :history_seq_len]
                    hidden_states_list[idx] = cur_hidden_states[:, history_seq_len:]

                    seq_start += cur_seq_len + history_seq_len

                hidden_states = torch.cat(hidden_states_list, dim=1)

                norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states,
                    navit_encoder_attention_mask,
                    None,
                    original_context_length,
                    original_context_length_list,
                    enable_navit,
                )
                hidden_states = hidden_states + attn_output

                seq_start = 0
                for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                    cur_hidden_states = hidden_states[:, seq_start : seq_start + cur_seq_len, :]

                    hidden_states_list[idx] = torch.cat([history_hidden_states_list[idx], cur_hidden_states], dim=1)

                    seq_start += cur_seq_len

                hidden_states = torch.cat(hidden_states_list, dim=1)
            else:
                history_hidden_states, hidden_states = (
                    hidden_states[:, :history_seq_len],
                    hidden_states[:, history_seq_len:],
                )
                norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states,
                    navit_encoder_attention_mask,
                    None,
                    original_context_length,
                    original_context_length_list,
                    enable_navit,
                )
                hidden_states = hidden_states + attn_output
                hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)
        else:
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states,
                navit_encoder_attention_mask,
                None,
                original_context_length,
                original_context_length_list,
                enable_navit,
            )
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

class GigaworldTransformer3DModelFunCtrl(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Gigaworld model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding",
        "patch_short",
        "patch_mid",
        "patch_long",
        "condition_embedder",
        "norm",
    ]
    _no_split_modules = ["GigaworldTransformerBlock", "GigaworldOutputNorm"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "history_key_scale",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["GigaworldTransformerBlock"]
    _cp_plan = {
        # Input split at attn level and ffn level.
        "blocks.*.attn1": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "rotary_emb": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*.attn2": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*.ffn": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        # Output gather at attn level and ffn level.
        **{f"blocks.{i}.attn1": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
        **{f"blocks.{i}.attn2": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
        **{f"blocks.{i}.ffn": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
    }

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_dim: tuple[int, ...] = (44, 42, 42),
        rope_theta: float = 10000.0,
        restrict_self_attn: bool = False,
        guidance_cross_attn: bool = False,
        is_train_restrict_lora: bool = False,
        restrict_lora: bool = False,
        restrict_lora_rank: int = 128,
        zero_history_timestep: bool = False,
        has_multi_term_memory_patch: bool = False,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",  # [scalar, per_head]
        is_use_gan: bool = False,
        is_use_gan_hooks: bool = False,
        is_use_gan_final: bool = False,
        gan_cond_map_dim: int = 768,
        gan_hooks: List[int] = [5, 15, 25, 35],
        model_type: str = "wan2.1",  # "wan2.1" | "wan2.2" | "wan2.2_5b"
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = GigaworldRotaryPosEmbed(rope_dim=rope_dim, theta=rope_theta)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        self.condition_embedder = GigaworldTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                GigaworldTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    restrict_self_attn=restrict_self_attn,
                    guidance_cross_attn=guidance_cross_attn,
                    is_train_restrict_lora=is_train_restrict_lora,
                    restrict_lora=restrict_lora,
                    restrict_lora_rank=restrict_lora_rank,
                    is_amplify_history=is_amplify_history,
                    history_scale_mode=history_scale_mode,
                )
                for _ in range(num_layers)
            ]
        )
        # 4. Output norm & projection
        self.norm_out = GigaworldOutputNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))

        self.init_weights()

        # 5. Initial Stage1
        self.zero_history_timestep = zero_history_timestep
        self.inner_dim = inner_dim
        self.model_type = model_type
        if has_multi_term_memory_patch:
            # NOTE: PATCH CH SIZE = VAE LATENT CH SIZE
            if self.model_type == "wan2.2_5b":
                memory_channels = 48
            elif self.model_type == "wan2.2" or self.model_type == "wan2.1":
                memory_channels = 16
            else: 
                raise ValueError(f"Unsupported model_type: {self.model_type}")
            self.patch_short = nn.Conv3d(memory_channels, self.inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.patch_mid = nn.Conv3d(memory_channels, self.inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
            self.patch_long = nn.Conv3d(memory_channels, self.inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            self.initialize_weight_from_another_conv3d(self.patch_embedding, memory_channels)

        # 6. Initial Gan
        self.is_use_gan = is_use_gan
        if is_use_gan:
            self.is_use_gan_hooks = is_use_gan_hooks
            self.is_use_gan_final = is_use_gan_final
            if is_use_gan_hooks:
                gan_heads = []
                self.gan_hooks = gan_hooks
                for hook in self.gan_hooks:
                    gan_heads.append((str(hook), Discriminator3DHead(inner_dim, gan_cond_map_dim)))
                self.gan_heads = nn.ModuleDict(gan_heads)
            if is_use_gan_final:
                self.gan_final_head = Discriminator3DHead(out_channels, gan_cond_map_dim)

        self.gradient_checkpointing = False

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer, in_channels):
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()
        weight = weight[:, :in_channels, :, :, :]

        sd = {
            "patch_short.weight": weight.clone(),
            "patch_short.bias": bias.clone(),
            "patch_mid.weight": einops.repeat(weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=2, hk=2, wk=2) / 8.0,
            "patch_mid.bias": bias.clone(),
            "patch_long.weight": einops.repeat(weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=4, hk=4, wk=4)
            / 64.0,
            "patch_long.bias": bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self.load_state_dict(sd, strict=False)

    def gradient_checkpointing_method(self, block, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            result = self._gradient_checkpointing_func(block, *args)
        else:
            result = block(*args)
        return result

    def enable_kv_cache(self):
        for block in self.blocks:
            if hasattr(block.attn1, "processor") and hasattr(block.attn1.processor, "enable_cache"):
                block.attn1.processor.enable_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            if hasattr(block.attn1, "processor") and hasattr(block.attn1.processor, "disable_cache"):
                block.attn1.processor.disable_cache()

    def clear_kv_cache(self):
        for block in self.blocks:
            if hasattr(block.attn1, "processor") and hasattr(block.attn1.processor, "clear_cache"):
                block.attn1.processor.clear_cache()

    def process_input_hidden_states(
        self,
        latents,
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
    ):
        height_list = []
        width_list = []
        temporal_list = []
        seq_list = []
        if isinstance(latents, list):
            hidden_states = None
            rope_freqs = None
            for idx, cur_hidden_states in enumerate(latents):
                cur_hidden_states = self.gradient_checkpointing_method(
                    self.patch_embedding, cur_hidden_states.to(self.device, dtype=self.dtype)
                )
                B, C, T, H, W = cur_hidden_states.shape

                cur_hidden_states = cur_hidden_states.flatten(2).transpose(1, 2)

                if indices_hidden_states is None:
                    indices_hidden_states = torch.arange(0, T).unsqueeze(0).expand(B, -1)

                cur_indices_latents = indices_hidden_states
                cur_rope_freqs = self.rope(
                    frame_indices=cur_indices_latents, height=H, width=W, device=cur_hidden_states.device
                )
                cur_rope_freqs = cur_rope_freqs.flatten(2).transpose(1, 2)

                height_list.append(H)
                width_list.append(W)
                temporal_list.append(T)
                seq_list.append(cur_hidden_states.shape[1])

                if hidden_states is None:
                    hidden_states = cur_hidden_states
                    rope_freqs = cur_rope_freqs
                else:
                    hidden_states = torch.cat([cur_hidden_states, hidden_states], dim=1)
                    rope_freqs = torch.cat([cur_rope_freqs, rope_freqs], dim=1)
        else:
            hidden_states = self.gradient_checkpointing_method(self.patch_embedding, latents)
            B, C, T, H, W = hidden_states.shape

            if indices_hidden_states is None:
                indices_hidden_states = torch.arange(0, T).unsqueeze(0).expand(B, -1)

            hidden_states = hidden_states.flatten(2).transpose(
                1, 2
            )  # torch.Size([1, 3072, 9, 44, 34]) -> torch.Size([1, 13464, 3072])

            rope_freqs = self.rope(
                frame_indices=indices_hidden_states,
                height=H,
                width=W,
                device=hidden_states.device,
            )  # torch.Size([1, 9]) -> torch.Size([1, 256, 9, 44, 34])
            rope_freqs = rope_freqs.flatten(2).transpose(1, 2)  # torch.Size([1, 13464, 256])

            height_list.append(H)
            width_list.append(W)
            temporal_list.append(T)
            seq_list.append(hidden_states.shape[1])

        # Process short history latents
        if latents_history_short is not None and indices_latents_history_short is not None:
            latents_history_short = latents_history_short.to(hidden_states)
            latents_history_short = self.gradient_checkpointing_method(self.patch_short, latents_history_short)
            _, _, _, H1, W1 = latents_history_short.shape
            latents_history_short = latents_history_short.flatten(2).transpose(1, 2)

            rope_freqs_history_short = self.rope(
                frame_indices=indices_latents_history_short,
                height=H1,
                width=W1,
                device=latents_history_short.device,
            )
            rope_freqs_history_short = rope_freqs_history_short.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_short, hidden_states], dim=1)
            rope_freqs = torch.cat([rope_freqs_history_short, rope_freqs], dim=1)

        # Process mid history latents
        if latents_history_mid is not None and indices_latents_history_mid is not None:
            latents_history_mid = latents_history_mid.to(hidden_states)
            latents_history_mid = pad_for_3d_conv(latents_history_mid, (2, 4, 4))
            latents_history_mid = self.gradient_checkpointing_method(self.patch_mid, latents_history_mid)
            latents_history_mid = latents_history_mid.flatten(2).transpose(1, 2)

            rope_freqs_history_mid = self.rope(
                frame_indices=indices_latents_history_mid,
                height=H1,
                width=W1,
                device=latents_history_mid.device,
            )
            rope_freqs_history_mid = pad_for_3d_conv(rope_freqs_history_mid, (2, 2, 2))
            rope_freqs_history_mid = center_down_sample_3d(rope_freqs_history_mid, (2, 2, 2))
            rope_freqs_history_mid = rope_freqs_history_mid.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_mid, hidden_states], dim=1)
            rope_freqs = torch.cat([rope_freqs_history_mid, rope_freqs], dim=1)

        # Process long history latents
        if latents_history_long is not None and indices_latents_history_long is not None:
            latents_history_long = latents_history_long.to(hidden_states)
            latents_history_long = pad_for_3d_conv(latents_history_long, (4, 8, 8))
            latents_history_long = self.gradient_checkpointing_method(self.patch_long, latents_history_long)
            latents_history_long = latents_history_long.flatten(2).transpose(1, 2)

            rope_freqs_history_long = self.rope(
                frame_indices=indices_latents_history_long,
                height=H1,
                width=W1,
                device=latents_history_long.device,
            )
            rope_freqs_history_long = pad_for_3d_conv(rope_freqs_history_long, (4, 4, 4))
            rope_freqs_history_long = center_down_sample_3d(rope_freqs_history_long, (4, 4, 4))
            rope_freqs_history_long = rope_freqs_history_long.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_long, hidden_states], dim=1)
            rope_freqs = torch.cat([rope_freqs_history_long, rope_freqs], dim=1)

        return (
            hidden_states,
            rope_freqs,
            height_list,
            width_list,
            temporal_list,
            seq_list,
        )

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        # ------------ Stage 1 ------------
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        is_first_denoising_step: bool = False,
        # ------------ GAN ------------
        gan_mode: bool = False,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        assert (
            len(
                {
                    x is None
                    for x in [
                        indices_hidden_states,
                        indices_latents_history_short,
                        indices_latents_history_mid,
                        indices_latents_history_long,
                        latents_history_short,
                        latents_history_mid,
                        latents_history_long,
                    ]
                }
            )
            == 1
        ), "All history latents and indices must either all exist or all be None"

        if indices_hidden_states is not None and indices_hidden_states.ndim == 1:
            indices_hidden_states = indices_hidden_states.unsqueeze(0)
        if indices_latents_history_short is not None and indices_latents_history_short.ndim == 1:
            indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        if indices_latents_history_mid is not None and indices_latents_history_mid.ndim == 1:
            indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        if indices_latents_history_long is not None and indices_latents_history_long.ndim == 1:
            indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

        if gan_mode:
            assert self.is_use_gan

        if isinstance(hidden_states, list):
            assert gan_mode is False and self.is_use_gan is False
            enable_navit = True
            navit_len = len(hidden_states)
            batch_size = hidden_states[0].shape[0]
        else:
            enable_navit = False
            batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size

        (
            hidden_states,
            rotary_emb,
            post_patch_height_list,
            post_patch_width_list,
            post_patch_num_frames_list,
            original_context_length_list,
        ) = self.process_input_hidden_states(
            latents=hidden_states,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
        )  # hidden: [high, mid, low] -> [low, mid, high]
        post_patch_num_frames = sum(post_patch_num_frames_list)
        post_patch_height = sum(post_patch_height_list)
        post_patch_width = sum(post_patch_width_list)
        original_context_length = sum(original_context_length_list)
        history_context_length = hidden_states.shape[1] - original_context_length

        if indices_hidden_states is not None and self.zero_history_timestep:
            if isinstance(timestep, list):
                timestep_t0 = torch.zeros((1), dtype=timestep[0].dtype, device=timestep[0].device)
            else:
                timestep_t0 = torch.zeros((1), dtype=timestep.dtype, device=timestep.device)
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        navit_hidden_attention_mask = None
        navit_encoder_attention_mask = None
        if enable_navit:
            assert navit_len == len(original_context_length_list)
            navit_hidden_attention_mask, navit_encoder_attention_mask, navit_history_hidden_attention_mask = (
                create_navit_attention_masks(
                    batch_size=batch_size,
                    original_context_length_list=original_context_length_list[::-1],
                    history_context_length=history_context_length,
                    encoder_hidden_states_seq_len=encoder_hidden_states.shape[1],
                    device=hidden_states.device,
                    restrict_self_attn=self.config.restrict_self_attn,
                    guidance_cross_attn=self.config.guidance_cross_attn,
                )
            )
            navit_hidden_attention_mask = [navit_hidden_attention_mask, navit_history_hidden_attention_mask]

            history_hidden_states, hidden_states = (
                hidden_states[:, :history_context_length],
                hidden_states[:, history_context_length:],
            )
            history_rotary_emb, rotary_emb = (
                rotary_emb[:, :history_context_length],
                rotary_emb[:, history_context_length:],
            )
            timestep = timestep[::-1]

            hidden_states_list = [None] * navit_len
            rotary_emb_list = [None] * navit_len
            temb_list = [None] * navit_len
            timestep_proj_list = [None] * navit_len

            seq_start = 0
            for idx, cur_seq_len in zip(range(navit_len), original_context_length_list[::-1]):
                cur_hidden_states = hidden_states[:, seq_start : seq_start + cur_seq_len, :]
                cur_rotary_emb = rotary_emb[:, seq_start : seq_start + cur_seq_len, :]

                hidden_states_list[idx] = torch.cat([history_hidden_states, cur_hidden_states], dim=1)
                rotary_emb_list[idx] = torch.cat([history_rotary_emb, cur_rotary_emb], dim=1)

                seq_start += cur_seq_len

                if idx == 0:
                    cur_temb, cur_timestep_proj, encoder_hidden_states = self.condition_embedder(
                        timestep[idx], encoder_hidden_states
                    )
                else:
                    cur_temb, cur_timestep_proj, _ = self.condition_embedder(
                        timestep[idx], encoder_hidden_states, is_return_encoder_hidden_states=False
                    )

                cur_temb = cur_temb.view(batch_size, 1, -1).expand(-1, cur_seq_len, -1)
                cur_timestep_proj = cur_timestep_proj.view(batch_size, 6, 1, -1).expand(-1, -1, cur_seq_len, -1)

                if self.zero_history_timestep:
                    temb_list[idx] = torch.cat([temb_t0, cur_temb], dim=1)
                    timestep_proj_list[idx] = torch.cat([timestep_proj_t0, cur_timestep_proj], dim=2)
                else:
                    temb_list[idx] = cur_temb
                    timestep_proj_list[idx] = cur_timestep_proj

            hidden_states = torch.cat(hidden_states_list, dim=1)
            rotary_emb = torch.cat(rotary_emb_list, dim=1)
            temb = torch.cat(temb_list, dim=1)
            timestep_proj = torch.cat(timestep_proj_list, dim=2)
        else:
            temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
            timestep_proj = timestep_proj.unflatten(-1, (6, -1))

            if indices_hidden_states is not None and not self.zero_history_timestep:
                main_repeat_size = hidden_states.shape[1]
            else:
                main_repeat_size = original_context_length
            temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
            timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, main_repeat_size, -1)

            if indices_hidden_states is not None and self.zero_history_timestep:
                temb = torch.cat([temb_t0, temb], dim=1)
                timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)

        # 4. Transformer blocks
        logits_hidden = []
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for iidx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])
        else:
            for iidx, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            if not enable_navit:
                temb = temb[:, -original_context_length:, :]
            shift, scale = (self.norm_out.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
                2, dim=2
            )
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.norm_out.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if enable_navit:
            hidden_states = (self.norm_out.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

            output = []
            seq_start = 0
            for (
                cur_original_context_length,
                cur_post_patch_num_frames,
                cur_post_patch_height,
                cur_post_patch_width,
            ) in zip(
                reversed(original_context_length_list),
                reversed(post_patch_num_frames_list),
                reversed(post_patch_height_list),
                reversed(post_patch_width_list),
            ):
                cur_hidden_states = hidden_states[
                    :, seq_start : seq_start + cur_original_context_length + history_context_length, :
                ]  # (B, T*H*W, C)
                cur_hidden_states = cur_hidden_states[:, history_context_length:, :]
                cur_hidden_states = self.proj_out(cur_hidden_states)
                seq_start += cur_original_context_length + history_context_length

                cur_hidden_states = cur_hidden_states.reshape(
                    batch_size,
                    cur_post_patch_num_frames,
                    cur_post_patch_height,
                    cur_post_patch_width,
                    p_t,
                    p_h,
                    p_w,
                    -1,
                )
                cur_hidden_states = cur_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                cur_hidden_states = cur_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                output.append(cur_hidden_states)

            output = output[::-1]
        else:
            hidden_states = hidden_states[:, -original_context_length:, :]
            hidden_states = (self.norm_out.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        logits = []
        if gan_mode and self.is_use_gan:
            if self.is_use_gan_final:
                logits.append(self.gradient_checkpointing_method(self.gan_final_head, output))
            if self.is_use_gan_hooks:
                for idx, (_, gan_head) in enumerate(self.gan_heads.items()):
                    activation = rearrange(
                        logits_hidden[idx],
                        "b (f h w) c -> b c f h w",
                        f=post_patch_num_frames,
                        h=post_patch_height,
                        w=post_patch_width,
                    )
                    logits.append(self.gradient_checkpointing_method(gan_head, activation.contiguous()))
            logits = torch.cat(logits, dim=1) if len(logits) > 1 else logits[0]
            logits_hidden = None
            del logits_hidden

        if not return_dict:
            return (output, logits)

        return Transformer2DModelOutput(sample=output, logits=logits)

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.condition_embedder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.proj_out.weight)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        subfolder=None,
        transformer_additional_kwargs={},
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
        device_map="cpu",
        max_workers=8,
        use_default_loader=False,
    ):
        if use_default_loader:
            return super().from_pretrained(
                pretrained_model_path, subfolder=subfolder, device_map=device_map, torch_dtype=torch_dtype
            )

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from huggingface_hub import snapshot_download

        from diffusers.utils import WEIGHTS_NAME

        if os.path.exists(pretrained_model_path):
            if subfolder is not None:
                pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            print(f"{_C['yellow']}Downloading from Hugging Face Hub: {pretrained_model_path}{_C['reset']}")
            cache_dir = snapshot_download(
                repo_id=pretrained_model_path,
                # allow_patterns=["*.json", "*.safetensors", "*.bin"],
            )
            pretrained_model_path = cache_dir
            if subfolder is not None:
                pretrained_model_path = os.path.join(cache_dir, subfolder)
        print(f"{_C['cyan']}loaded 3D transformer's pretrained weights from {pretrained_model_path} ...{_C['reset']}")

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        def remap_state_dict_keys(state_dict):
            """Remap old key names to new key names for compatibility."""
            remapped = {}
            for key, value in state_dict.items():
                new_key = key
                # Only remap top-level scale_shift_table, not blocks.*.scale_shift_table
                if key == "scale_shift_table":
                    new_key = "norm_out.scale_shift_table"
                    print(f"{_C['dim']}Remapping key: {key} -> {new_key}{_C['reset']}")
                remapped[new_key] = value
            return remapped

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.model_loading_utils import load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available

                if is_accelerate_available():
                    import accelerate

                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file

                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file

                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(f"{_C['dim']}Loading {len(model_files_safetensors)} safetensors files with {max_workers} workers...{_C['reset']}")
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_file = {executor.submit(load_file, f): f for f in model_files_safetensors}
                        for future in as_completed(future_to_file):
                            _state_dict = future.result()
                            state_dict.update(_state_dict)

                # Remap keys before loading into meta model
                state_dict = remap_state_dict_keys(state_dict)

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                        keep_in_fp32_modules=cls._keep_in_fp32_modules,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"{_C['yellow']}Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}{_C['reset']}"
                        )

                return model
            except Exception as e:
                print(f"{_C['yellow']}The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead.{_C['reset']}")

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file

            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file

            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            print(f"{_C['dim']}Loading {len(model_files_safetensors)} safetensors files with {max_workers} workers...{_C['reset']}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(load_file, f): f for f in model_files_safetensors}
                for future in as_completed(future_to_file):
                    _state_dict = future.result()
                    state_dict.update(_state_dict)

        # Remap keys before size check and loading
        state_dict = remap_state_dict_keys(state_dict)

        tmp_state_dict = {}
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(f"{_C['yellow']}{key} Size don't match, skip{_C['reset']}")

        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        m_color = _C['yellow'] if len(m) > 0 else _C['green']
        u_color = _C['yellow'] if len(u) > 0 else _C['green']
        print(f"{m_color}### missing keys: {len(m)};{_C['reset']}\n{u_color}### unexpected keys: {len(u)};{_C['reset']}")
        print(f"{_C['dim']}{m}{_C['reset']}")

        for name, param in model.named_parameters():
            should_keep_fp32 = any(pattern in name for pattern in cls._keep_in_fp32_modules)
            if should_keep_fp32:
                param.data = param.data.to(torch.float32)
                # print(f"Keeping parameter {name} in fp32")
            else:
                param.data = param.data.to(torch_dtype)
        model = model.to(device_map)

        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"{_C['blue']}### All Parameters: {sum(params) / 1e6:.3f} M{_C['reset']}")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"{_C['magenta']}### attn1 Parameters: {sum(params) / 1e6:.3f} M{_C['reset']}")

        params = [p.numel() if "attn2." in n else 0 for n, p in model.named_parameters()]
        print(f"{_C['magenta']}### attn2 Parameters: {sum(params) / 1e6:.3f} M{_C['reset']}")

        return model


if __name__ == "__main__":
    import os

    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
    os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"
    # export DIFFUSERS_ENABLE_HUB_KERNELS=yes

    # def compare_models(model1, model2):
    #     for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
    #         if name1 != name2:
    #             print(f"参数名不同: {name1} vs {name2}")
    #             return False
    #         if not torch.equal(param1, param2):
    #             print(f"参数 {name1} 的值不同")
    #             print(f"最大差异: {torch.max(torch.abs(param1 - param2))}")
    #             return False
    #     print("所有参数完全相同！")
    #     return True
    # compare_models(transformer, transformer1)

    gan_mode = False
    is_use_gan_hooks = False
    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "guidance_cross_attn": True,
        "restrict_self_attn": False,
        "restrict_lora": False,
        "is_train_restrict_lora": False,
        "is_amplify_history": False,
        "history_scale_mode": "per_head",  # [scalar, per_head]
        "is_use_gan": gan_mode,
        "is_use_gan_hooks": is_use_gan_hooks,
        "gan_hooks": [13, 21, 29],
        "gan_cond_map_dim": 768,
        # "gan_hooks": [10, 20, 30],
        # "gan_cond_map_dim": 512,
    }
    # transformer_additional_kwargs={}

    device = "cuda"
    weight_dtype = torch.bfloat16
    transformer = GigaworldTransformer3DModel.from_pretrained(
        "/mnt/pfs/users/zhanqian.wu/ckpt/Wan2.1-Fun-V1.1-1.3B-Control-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        transformer_additional_kwargs=transformer_additional_kwargs,
    )
    transformer.requires_grad_(False)
    transformer.eval()
    transformer = transformer.to(device, dtype=weight_dtype)

    # import sys
    # from argparse import Namespace
    # sys.path.append("../../")
    # from gigaworld.utils.utils_gigaworld_base import save_extra_components, load_extra_components
    # args = Namespace()
    # args.training_config = Namespace()
    # args.training_config.is_enable_stage1 = True
    # args.training_config.is_train_restrict_lora = True
    # save_extra_components(args, transformer, "./temp")
    # load_extra_components(args, transformer, "./temp/transformer_partial.pth")

    is_navit = False
    batch_size = 4
    max_length = 512
    if is_navit:
        noisy_model_input = [
            torch.randn(batch_size, 48, 9, 12, 20),
            torch.randn(batch_size, 48, 9, 24, 40),
            torch.randn(batch_size, 48, 9, 48, 80),
        ]
        timesteps = [
            torch.randint(0, 1000, (batch_size,)).to(device),
            torch.randint(0, 1000, (batch_size,)).to(device),
            torch.randint(0, 1000, (batch_size,)).to(device),
        ]
    else:
        noisy_model_input = torch.randn(batch_size, 48, 9, 48, 80).to(device, dtype=weight_dtype)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

    prompt_embeds = torch.randn(batch_size, max_length, 4096).to(device, dtype=weight_dtype)
    indices_hidden_states = torch.randint(0, 10, (batch_size, 9)).to(device)
    indices_latents_history_short = torch.randint(0, 3, (batch_size, 2)).to(device)
    indices_latents_history_mid = torch.randint(0, 3, (batch_size, 2)).to(device)
    indices_latents_history_long = torch.randint(0, 17, (batch_size, 16)).to(device)
    latents_history_short = torch.randn(batch_size, 16, 2, 48, 80).to(device, dtype=weight_dtype)
    latents_history_mid = torch.randn(batch_size, 16, 2, 48, 80).to(device, dtype=weight_dtype)
    latents_history_long = torch.randn(batch_size, 16, 16, 48, 80).to(device, dtype=weight_dtype)

    # 16 2 2: 2400
    # 16 2 3: 3360
    # 16 4 2: 2640
    # 16 4 3: 3600
    #  8 2 2: 2280
    #  8 2 3: 3240

    # noisy_model_input_1 = torch.randn(batch_size, 16, 9, 12, 20).to(device, dtype=weight_dtype)
    # timesteps_1 = torch.randint(0, 1000, (batch_size,)).to(device)
    # noisy_model_input = [noisy_model_input_1, noisy_model_input_1, noisy_model_input_1]
    # timesteps = [timesteps_1, timesteps_1, torch.randint(0, 1000, (batch_size,)).to(device)]

    model_pred = transformer(
        hidden_states=noisy_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=latents_history_short.to(weight_dtype),
        latents_history_mid=latents_history_mid.to(weight_dtype),
        latents_history_long=latents_history_long.to(weight_dtype),
        gan_mode=gan_mode,
        return_dict=False,
    )[0]
