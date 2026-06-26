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
from radial_attn.models.wan.inference import replace_wan_attention
from radial_attn.models.wan.sparse_transformer import replace_sparse_forward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers", help="Model ID to use for generation")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")
    parser.add_argument("--height", type=int, default=768, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=69, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--flow_shift", type=float, default=5.0, help="Flow shift for the scheduler, adjust based on video resolution (5.0 for 720P, 3.0 for 480P)")
    parser.add_argument("--lora_checkpoint_dir", type=str, default=None, help="Directory containing LoRA checkpoint files")
    parser.add_argument("--lora_checkpoint_name", type=str, default=None, help="Name of the LoRA checkpoint file to load, if not specified, will use the default name in the directory")
    parser.add_argument("--pattern", type=str, default="dense", choices=["radial", "dense"])
    parser.add_argument("--dense_layers", type=int, default=0, help="Number of dense layers to use in the Wan attention, set to 1 for 1x length video, 2 for 2x length video")
    parser.add_argument("--dense_timesteps", type=int, default=0, help="Number of dense timesteps to use in the Wan attention, set to 12 for 1x length video, 2 for 2x length video")
    parser.add_argument("--decay_factor", type=float, default=1, help="Decay factor for the Wan attention, we use this to control window width")
    parser.add_argument("--use_sage_attention", action="store_true", help="Use SAGE attention for quantized inference")
    parser.add_argument("--use_model_offload", action="store_true", help="Enable model offloading to CPU for memory efficiency")
    # Parallel inference parameters
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
    
    replace_sparse_forward()
    
    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = args.flow_shift
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    
    if args.use_model_offload:
        print("Using model offloading for memory efficiency")
        apply_group_offloading(text_encoder,
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=4
        )
        transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
        )

    pipe = WanPipeline.from_pretrained(model_id, text_encoder=text_encoder, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    if args.lora_checkpoint_dir is not None:
        pipe.load_lora_weights(
            args.lora_checkpoint_dir,
            weight_name=args.lora_checkpoint_name,
        )
        
    pipe.to("cuda")

    if args.prompt is None:
        print(colored("Using default prompt", "red"))
        args.prompt = "A cat walks on the grass, realistic"
    
    if args.negative_prompt is None:
        args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    print("=" * 20 + " Prompts " + "=" * 20)
    print(f"Prompt: {args.prompt}\n\n" + f"Negative Prompt: {args.negative_prompt}")

    if args.pattern == "radial":
        replace_wan_attention(
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
        num_inference_steps=args.num_inference_steps
    ).frames[0]
    
    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    export_to_video(output, args.output_file, fps=16)