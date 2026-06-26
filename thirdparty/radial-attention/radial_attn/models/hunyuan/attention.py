import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from ...attn_mask import RadialAttention
from typing import Optional
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.distributed as dist

try:
    from xfuser.core.distributed import get_ulysses_parallel_world_size
    from xfuser.model_executor.layers.usp import _ft_c_input_all_to_all, _ft_c_output_all_to_all
except:
    pass

class HunyuanVideoAttnSparseProcessor2_0:
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"  # default to radial attention, can be changed to "dense"
    use_sage_attention = False
    
    def __init__(self, layer_idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
            
        self.layer_idx = layer_idx
        self.use_sp = False

        if dist.is_initialized() and get_ulysses_parallel_world_size() > 1:
            self.use_sp = True

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        numeral_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        # print(f"numeral_timestep: {numeral_timestep}, dense_timestep: {self.dense_timestep}, layer_idx: {self.layer_idx}, dense_block: {self.dense_block}, sparse_type: {self.sparse_type}")

        if self.use_sp:
            # input qkv ulysses all_to_all comm
            text_seq_length = encoder_hidden_states.size(1)
            # Ugly but useful for MMDiT. TODO: handle layout inside all_to_all for cleaner code
            # for sparse attention,the layout of sequence must be [video_1, video_2, ..., text_1, text_2, ...],
            # [video_1, text_1, video_2, text_2, ...] will lead to different attention map
            query_text = query[:, :, -text_seq_length:, :]
            query_video = query[:, :, :-text_seq_length, :]
            query_text = _ft_c_input_all_to_all(query_text)
            query_video = _ft_c_input_all_to_all(query_video)
            query = torch.cat([query_video, query_text], dim=-2)

            key_text = key[:, :, -text_seq_length:, :]
            key_video = key[:, :, :-text_seq_length, :]
            key_text = _ft_c_input_all_to_all(key_text)
            key_video = _ft_c_input_all_to_all(key_video)
            key = torch.cat([key_video, key_text], dim=-2)

            value_text = value[:, :, -text_seq_length:, :]
            value_video = value[:, :, :-text_seq_length, :]
            value_text = _ft_c_input_all_to_all(value_text)
            value_video = _ft_c_input_all_to_all(value_video)
            value = torch.cat([value_video, value_text], dim=-2)

        pre_defined_mask = attention_mask[0, 0].expand(query.shape[2], query.shape[2])
        batch_size = query.shape[0]
        query = rearrange(query, "b h s d" " -> (b s) h d").contiguous()
        key = rearrange(key, "b h s d" " -> (b s) h d").contiguous()
        value = rearrange(value, "b h s d" " -> (b s) h d").contiguous()
        if timestep is None or numeral_timestep < self.dense_timestep or self.layer_idx < self.dense_block or self.sparse_type == "dense":
            # apply dense attention
            hidden_states = RadialAttention(
                query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type="dense", block_size=128, decay_factor=self.decay_factor, model_type="hunyuan", pre_defined_mask=pre_defined_mask, use_sage_attention=self.use_sage_attention
            )
        else:
            # apply radial attention
            hidden_states = RadialAttention(
                query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type=self.sparse_type, block_size=128, decay_factor=self.decay_factor, model_type="hunyuan", pre_defined_mask=pre_defined_mask, use_sage_attention=self.use_sage_attention
            )
            
        hidden_states = rearrange(hidden_states, "(b s) h d -> b h s d", b=batch_size)

        if self.use_sp:
            # output o ulysses all_to_all comm
            out = hidden_states
            out_text = out[:, :, -get_ulysses_parallel_world_size() * text_seq_length:, :]
            out_latents = out[:, :, : -get_ulysses_parallel_world_size() * text_seq_length, :]
            out_text = _ft_c_output_all_to_all(out_text)
            out_latents = _ft_c_output_all_to_all(out_latents)
            hidden_states = torch.cat([out_latents, out_text], dim=-2)

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
