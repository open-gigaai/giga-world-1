import os
import json
from termcolor import colored

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import UMT5EncoderModel
import argparse

from radial_attn.utils import set_seed
from radial_attn.models.wan2_2.inference import replace_wan22_attention
from radial_attn.models.wan2_2.sparse_transformer import replace_wan22_sparse_forward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan2.2-Diffuser")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers", help="Model ID to use for generation")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of denoising steps in the generated video")
    parser.add_argument("--output_file", type=str, default="wan22_output.mp4", help="Output video file name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--guidance_scale_2", type=float, default=3.0, help="Guidance scale for the second transformer (Wan2.2)")
    parser.add_argument("--lora_checkpoint_dir", type=str, default=None, help="Directory containing LoRA checkpoint files")
    parser.add_argument("--lora_checkpoint_name", type=str, default=None, help="Name of the LoRA checkpoint file to load, if not specified, will use the default name in the directory")
    parser.add_argument("--pattern", type=str, default="dense", choices=["radial", "dense"])
    parser.add_argument("--dense_layers", type=int, default=0, help="Number of dense layers to use in the Wan attention")
    parser.add_argument("--dense_timesteps", type=int, default=0, help="Number of dense timesteps to use in the Wan attention")
    parser.add_argument("--decay_factor", type=float, default=1, help="Decay factor for the Wan attention, we use this to control window width")
    parser.add_argument("--use_sage_attention", action="store_true", help="Use SAGE attention for quantized inference")
    parser.add_argument("--use_model_offload", action="store_true", help="Enable model offloading to CPU for memory efficiency")
    parser.add_argument("--use_sequence_parallel", action="store_true",default=False, help="Enable sequence parallelism for parallel inference")
    parser.add_argument("--ulysses_degree", type=int, default=1, help="The number of ulysses parallel")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.use_sequence_parallel:
        import torch.distributed as dist
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank

        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)

            if args.ulysses_degree > 1 and world_size == args.ulysses_degree:
                from xfuser.core.distributed import (
                    init_distributed_environment,
                    initialize_model_parallel,
                )

                init_distributed_environment(
                    rank=dist.get_rank(), world_size=dist.get_world_size())

                initialize_model_parallel(
                    sequence_parallel_degree=dist.get_world_size(),
                    ring_degree=1,
                    ulysses_degree=args.ulysses_degree,
                )
            else:
                assert "only ulysses parallelism is supported now"
        else:
            assert "parallel world_size must bigger than 1"
    
    replace_wan22_sparse_forward()
    
    # Load Wan2.2 model components (following wan2_2_test.py initialization)
    model_id = args.model_id
    
    # Load VAE with tiling enabled
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    vae.enable_tiling()
    
    # Load pipeline
    if args.use_model_offload:
        # Load components separately for model offloading
        transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        transformer_2 = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer_2", torch_dtype=torch.bfloat16)
        text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)

        # Apply group offloading to each component
        transformer.enable_group_offload(torch.device("cuda"), torch.device("cpu"), "block_level", num_blocks_per_group=5)
        transformer_2.enable_group_offload(torch.device("cuda"), torch.device("cpu"), "block_level", num_blocks_per_group=5)
        apply_group_offloading(text_encoder, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="block_level", num_blocks_per_group=4)

        # Create pipeline with offloaded components
        pipe = WanPipeline.from_pretrained(model_id, transformer=transformer, transformer_2=transformer_2, text_encoder=text_encoder, vae=vae, torch_dtype=torch.bfloat16)
    else:
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

    if args.lora_checkpoint_dir is not None:
        pipe.load_lora_weights(
            args.lora_checkpoint_dir,
            weight_name=args.lora_checkpoint_name,
        )

    pipe.to("cuda")

    if args.prompt is None:
        print(colored("Using default prompt", "red"))
        args.prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    
    if args.negative_prompt is None:
        args.negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    print("=" * 20 + " Prompts " + "=" * 20)
    print(f"Prompt: {args.prompt}\n\n" + f"Negative Prompt: {args.negative_prompt}")

    if args.pattern == "radial":
        replace_wan22_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            args.dense_layers,
            args.dense_timesteps,
            args.decay_factor,
            args.pattern,
            args.use_sage_attention,
        )
        
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=args.num_inference_steps
    ).frames[0]
    
    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    export_to_video(output, args.output_file, fps=16, backend="imageio")