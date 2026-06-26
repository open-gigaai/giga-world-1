import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import AttentionModuleMixin
from .attention import Wan22SparseAttnProcessor 
from .sparse_transformer import replace_wan22_sparse_forward
from ...attn_mask import MaskMap

def replace_wan22_attention(
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
    """
    Replace Wan2.2 model self-attention (attn1) with radial sparse attention.
    Cross-attention (attn2) remains unchanged.
    
    Args:
        pipe: WanPipeline instance
        height: Video height
        width: Video width 
        num_frames: Number of frames
        dense_layers: Number of initial layers to keep dense
        dense_timesteps: Number of initial timesteps to keep dense
        decay_factor: Decay factor for radial attention
        sparsity_type: Type of sparsity pattern ("radial" or "dense")
        use_sage_attention: Whether to use SageAttention backend
    """
    
    # Calculate token dimensions based on transformer configuration
    transformer = pipe.transformer if pipe.transformer is not None else pipe.transformer_2
    
    # Account for temporal and spatial patch sizes
    temporal_patch_size = transformer.config.patch_size[0]
    spatial_patch_size = transformer.config.patch_size[1]  # Assuming square patches
    
    # Calculate latent dimensions after patching
    num_latent_frames = 1 + (num_frames - 1) // (pipe.vae_scale_factor_temporal * temporal_patch_size)
    latent_height = height // (pipe.vae_scale_factor_spatial * spatial_patch_size)
    latent_width = width // (pipe.vae_scale_factor_spatial * spatial_patch_size)
    
    frame_size = latent_height * latent_width
    total_tokens = frame_size * num_latent_frames
    
    AttnModule = Wan22SparseAttnProcessor
    AttnModule.dense_block = dense_layers
    AttnModule.dense_timestep = dense_timesteps
    AttnModule.mask_map = MaskMap(video_token_num=total_tokens, num_frame=num_latent_frames)
    AttnModule.decay_factor = decay_factor
    AttnModule.sparse_type = sparsity_type
    AttnModule.use_sage_attention = use_sage_attention
    
    print(f"Replacing Wan2.2 self-attention with {sparsity_type} attention")
    print(f"Video config: {num_frames} frames -> {num_latent_frames} latent frames")
    print(f"Spatial config: {height}x{width} -> {latent_height}x{latent_width} latents")
    print(f"Total tokens: {total_tokens}, Frame size: {frame_size}")
    print(f"Dense layers: {dense_layers}, Dense timesteps: {dense_timesteps}")
    print(f"Decay factor: {decay_factor}, Use SageAttention: {use_sage_attention}")
    
    # Replace attention processors for primary transformer
    if pipe.transformer is not None:
        for layer_idx, block in enumerate(pipe.transformer.blocks):
            # Only set layer index for self-attention (attn1)
            if hasattr(block.attn1, 'processor'):
                block.attn1.processor.layer_idx = layer_idx
                # Replace only self-attention with sparse processor
                block.attn1.set_processor(AttnModule(layer_idx))
    
    # Replace attention processors for secondary transformer (if exists)
    if pipe.transformer_2 is not None:
        for layer_idx, block in enumerate(pipe.transformer_2.blocks):
            # Only set layer index for self-attention (attn1)
            if hasattr(block.attn1, 'processor'):
                block.attn1.processor.layer_idx = layer_idx
                # Replace only self-attention with sparse processor
                block.attn1.set_processor(AttnModule(layer_idx))
    
    # Replace forward methods with sparse versions
    replace_wan22_sparse_forward()
    
    print("Successfully replaced Wan2.2 self-attention processors (attn1 only)")

def setup_wan22_radial_attention(
    pipe,
    height=720,
    width=1280,
    num_frames=49,
    dense_layers=2,
    dense_timesteps=5,
    decay_factor=0.8,
    use_sage_attention=False,
):
    """
    Convenience function to setup radial attention for Wan2.2 with common parameters.
    
    Args:
        pipe: WanPipeline instance
        height: Video height (default: 720)
        width: Video width (default: 1280)
        num_frames: Number of frames (default: 49)
        dense_layers: Dense layers at start (default: 2)
        dense_timesteps: Dense timesteps at start (default: 5)
        decay_factor: Radial decay factor (default: 0.8)
        use_sage_attention: Use SageAttention backend (default: False)
    """
    replace_wan22_attention(
        pipe=pipe,
        height=height,
        width=width,
        num_frames=num_frames,
        dense_layers=dense_layers,
        dense_timesteps=dense_timesteps,
        decay_factor=decay_factor,
        sparsity_type="radial",
        use_sage_attention=use_sage_attention,
    )
    
def disable_wan22_radial_attention(pipe):
    """
    Disable radial attention and revert to dense attention.
    
    Args:
        pipe: WanPipeline instance
    """
    replace_wan22_attention(
        pipe=pipe,
        height=720,  # dummy values, won't affect dense attention
        width=1280,
        num_frames=49,
        dense_layers=0,
        dense_timesteps=0,
        decay_factor=1.0,
        sparsity_type="dense",
        use_sage_attention=False,
    )