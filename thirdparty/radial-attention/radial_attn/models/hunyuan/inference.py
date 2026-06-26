import torch
from diffusers.models.attention_processor import Attention
from .attention import HunyuanVideoAttnSparseProcessor2_0
from .sparse_transformer import replace_sparse_forward
from ...attn_mask import MaskMap

def replace_hunyuan_attention(
    pipe,
    height,
    width,
    num_frames,
    dense_layers=0,
    dense_timesteps=0,
    decay_factor=1.0,
    sparsity_type="radial",
    use_sage_attention=False,
):
    num_frames = 1 + (num_frames - 1) // (pipe.vae_scale_factor_temporal)
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size
    frame_size = int(height // mod_value) * int(width // mod_value)

    AttnModule = HunyuanVideoAttnSparseProcessor2_0
    AttnModule.dense_block = dense_layers
    AttnModule.dense_timestep = dense_timesteps
    AttnModule.mask_map = MaskMap(video_token_num=frame_size * num_frames, num_frame=num_frames)
    AttnModule.decay_factor = decay_factor
    AttnModule.sparse_type = sparsity_type
    AttnModule.use_sage_attention = use_sage_attention

    print(f"Replacing Hunyuan attention with {sparsity_type} attention")
    print(f"video token num: {AttnModule.mask_map.video_token_num}, num frames: {num_frames}")
    print(f"dense layers: {dense_layers}, dense timesteps: {dense_timesteps}, decay factor: {decay_factor}")

    for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
        m.attn.processor.layer_idx = layer_idx

    for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
        m.attn.processor.layer_idx = layer_idx + 20
        
    for _, m in pipe.transformer.named_modules():
        if isinstance(m, Attention) and hasattr(m.processor, "layer_idx"):
            layer_idx = m.processor.layer_idx
            m.set_processor(AttnModule(layer_idx))