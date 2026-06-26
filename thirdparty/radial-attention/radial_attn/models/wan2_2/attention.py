from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import WanAttention, _get_qkv_projections, _get_added_kv_projections
from einops import rearrange
from ...attn_mask import RadialAttention
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.distributed as dist

try:
    from xfuser.core.distributed import get_ulysses_parallel_world_size
    from xfuser.model_executor.layers.usp import _ft_c_input_all_to_all, _ft_c_output_all_to_all
except:
    pass

class Wan22SparseAttnProcessor:
    """
    Radial attention processor for Wan2.2 model with support for expand_timesteps feature.
    """
    _attention_backend = None
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"
    use_sage_attention = False

    def __init__(self, layer_idx: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "Wan22SparseAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        self.layer_idx = layer_idx
        self.use_sp = False

        if dist.is_initialized() and get_ulysses_parallel_world_size() > 1:
            self.use_sp = True

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        numeral_timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
            
        if attn.cross_attention_dim_head is not None: # case for cross attention
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                
        else: # case for sparse attention
            
            # Handle both scalar and tensor numeral_timestep for wan2.2 compatibility
            timestep_value = numeral_timestep
            if torch.is_tensor(numeral_timestep):
                timestep_value = numeral_timestep.item() if numeral_timestep.numel() == 1 else numeral_timestep[0].item()
            
            if timestep_value < self.dense_timestep or self.layer_idx < self.dense_block or self.sparse_type == "dense":
                batch_size = query.shape[0]
                
                if self.use_sp:
                    batch_size = query.shape[0]
                    # Ugly but useful now. TODO: modify all_to_all fuc of xdit to handle different layouts
                    query = rearrange(query, "b s h d" " -> b h s d").contiguous()
                    key = rearrange(key, "b s h d" " -> b h s d").contiguous()
                    value = rearrange(value, "b s h d" " -> b h s d").contiguous()
                    
                    # input all_to_all comm needs [b h s d] layout
                    query = _ft_c_input_all_to_all(query)
                    key = _ft_c_input_all_to_all(key)
                    value = _ft_c_input_all_to_all(value)

                    query = rearrange(query, "b h s d" " -> (b s) h d").contiguous()
                    key = rearrange(key, "b h s d" " -> (b s) h d").contiguous()
                    value = rearrange(value, "b h s d" " -> (b s) h d").contiguous()
                else:
                    query = rearrange(query, "b s h d -> (b s) h d")
                    key = rearrange(key, "b s h d -> (b s) h d")
                    value = rearrange(value, "b s h d -> (b s) h d")
                
                hidden_states = RadialAttention(
                    query=query, key=key, value=value,
                    mask_map=self.mask_map, sparsity_type="dense", block_size=64, decay_factor=self.decay_factor, model_type="wan", pre_defined_mask=None, use_sage_attention=self.use_sage_attention
                )

                if self.use_sp:
                    hidden_states = rearrange(hidden_states.contiguous(), "(b s) h d -> b h s d", b=batch_size).contiguous()
                    # output all_to_all comm needs [b h s d] layout
                    hidden_states = _ft_c_output_all_to_all(hidden_states)
                    hidden_states = rearrange(hidden_states, "b h s d -> b s h d", b=batch_size).contiguous()
                else:
                    hidden_states = rearrange(hidden_states, "(b s) h d -> b s h d", b=batch_size)
            else:
                batch_size = query.shape[0]
                if self.use_sp:
                    # Ugly but useful now. TODO: modify all_to_all fuc of xdit to handle different layouts
                    query = rearrange(query, "b s h d" " -> b h s d").contiguous()
                    key = rearrange(key, "b s h d" " -> b h s d").contiguous()
                    value = rearrange(value, "b s h d" " -> b h s d").contiguous()
                    
                    # input all_to_all comm needs [b h s d] layout
                    query = _ft_c_input_all_to_all(query)
                    key = _ft_c_input_all_to_all(key)
                    value = _ft_c_input_all_to_all(value)

                    query = rearrange(query, "b h s d -> (b s) h d").contiguous()
                    key = rearrange(key, "b h s d -> (b s) h d").contiguous()
                    value = rearrange(value, "b h s d -> (b s) h d").contiguous()
                else:
                    query = rearrange(query, "b s h d -> (b s) h d")
                    key = rearrange(key, "b s h d -> (b s) h d")
                    value = rearrange(value, "b s h d -> (b s) h d")
                
                # apply radial attention
                hidden_states = RadialAttention(
                    query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type="radial", block_size=64, decay_factor=self.decay_factor, model_type="wan", pre_defined_mask=None, use_sage_attention=self.use_sage_attention
                )

                if self.use_sp:
                    hidden_states = rearrange(hidden_states.contiguous(), "(b s) h d -> b h s d", b=batch_size).contiguous()
                    # output all_to_all comm needs [b h s d] layout
                    hidden_states = _ft_c_output_all_to_all(hidden_states)
                    hidden_states = rearrange(hidden_states, "b h s d -> b s h d", b=batch_size).contiguous()
                else:
                    hidden_states = rearrange(hidden_states, "(b s) h d -> b s h d", b=batch_size)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class Wan22SparseAttnProcessor2_0:
    """
    Alternative processor for Wan2.2 with different tensor reshaping approach.
    """
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"
    use_sage_attention = False
    
    def __init__(self, layer_idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Wan22SparseAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.use_sp = False

        if dist.is_initialized() and get_ulysses_parallel_world_size() > 1:
            self.use_sp = True
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        numeral_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
            
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                x1, x2 = x[..., 0], x[..., 1]
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
        
        if timestep is None: # this is the case for dense attention or cross attention
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
        else: # this is the case for sparse attention
            batch_size = query.shape[0]
            if self.use_sp:
                batch_size = query.shape[0]
                # Ugly but useful now. TODO: modify all_to_all fuc of xdit to handle different layouts
                query = rearrange(query, "b s h d" " -> b h s d").contiguous()
                key = rearrange(key, "b s h d" " -> b h s d").contiguous()
                value = rearrange(value, "b s h d" " -> b h s d").contiguous()
                
                # input all_to_all comm needs [b h s d] layout
                query = _ft_c_input_all_to_all(query)
                key = _ft_c_input_all_to_all(key)
                value = _ft_c_input_all_to_all(value)

                query = rearrange(query, "b h s d -> (b s) h d").contiguous()
                key = rearrange(key, "b h s d -> (b s) h d").contiguous()
                value = rearrange(value, "b h s d -> (b s) h d").contiguous()
            else:
                query = rearrange(query, "b s h d -> (b s) h d")
                key = rearrange(key, "b s h d -> (b s) h d")
                value = rearrange(value, "b s h d -> (b s) h d")
            
            # Handle both scalar and tensor numeral_timestep for wan2.2 compatibility
            timestep_value = numeral_timestep
            if torch.is_tensor(numeral_timestep):
                timestep_value = numeral_timestep.item() if numeral_timestep.numel() == 1 else numeral_timestep[0].item()
            
            if timestep_value < self.dense_timestep or self.layer_idx < self.dense_block or self.sparse_type == "dense":
                hidden_states = RadialAttention(
                    query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type="dense", block_size=128, decay_factor=self.decay_factor, model_type="wan", pre_defined_mask=None, use_sage_attention=self.use_sage_attention
                )
            else:
                # apply radial attention
                hidden_states = RadialAttention(
                    query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type="radial", block_size=128, decay_factor=self.decay_factor, model_type="wan", pre_defined_mask=None, use_sage_attention=self.use_sage_attention
                )
                
            if self.use_sp:
                hidden_states = rearrange(hidden_states.contiguous(), "(b s) h d -> b h s d", b=batch_size).contiguous()
                # output all_to_all comm needs [b h s d] layout
                hidden_states = _ft_c_output_all_to_all(hidden_states)
                hidden_states = rearrange(hidden_states, "b h s d -> b s h d", b=batch_size).contiguous()
            else:
                hidden_states = rearrange(hidden_states, "(b s) h d -> b s h d", b=batch_size)
           
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states