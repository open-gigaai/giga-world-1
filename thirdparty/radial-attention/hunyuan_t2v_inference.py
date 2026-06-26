import os
import json
from termcolor import colored
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video
import argparse
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from radial_attn.models.hunyuan.sparse_transformer import replace_sparse_forward

from radial_attn.utils import set_seed
from radial_attn.models.hunyuan.inference import replace_hunyuan_attention

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using HunyuanVideo with custom attention")
    parser.add_argument("--model_id", type=str, default="hunyuanvideo-community/HunyuanVideo", 
                       help="Model ID to use for generation")
    parser.add_argument("--prompt", type=str, default=None, 
                       help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, 
                       help="Negative text prompt to avoid certain features")
    parser.add_argument("--height", type=int, default=768, 
                       help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, 
                       help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=117, 
                       help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, 
                       help="Number of denoising steps in the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", 
                       help="Output video file name")
    parser.add_argument("--seed", type=int, default=0, 
                       help="Random seed for generation")

    # Custom attention parameters
    parser.add_argument("--pattern", type=str, default="dense", 
                       choices=["radial", "dense"])
    parser.add_argument("--dense_layers", type=int, default=0, 
                       help="Number of dense layers to use in the attention")
    parser.add_argument("--dense_timesteps", type=int, default=0, 
                       help="Number of dense timesteps to use in the attention")
    parser.add_argument("--decay_factor", type=float, default=1, 
                       help="Decay factor for the attention, controls window width")
    
    parser.add_argument("--lora_checkpoint_dir", type=str, default=None, 
                       help="Directory containing LoRA checkpoint files")
    
    parser.add_argument("--use_sage_attention", action="store_true",
                        help="Use SAGE attention for quantized inference")
    
    parser.add_argument("--use_model_offload", action="store_true",
                        help="Enable model offloading to CPU for memory efficiency")

    # Parallel inference parameters
    parser.add_argument("--use_sequence_parallel", action="store_true",
                                help="Enable sequence parallelism for parallel inference")
    parser.add_argument("--ulysses_degree", type=int, default=2,
                                help="The number of ulysses parallel")

    args = parser.parse_args()

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

    set_seed(args.seed)
    
    replace_sparse_forward()
    
    # Load model with bfloat16 precision
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id,  
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    
    if args.use_model_offload:
        print("Using model offloading for memory efficiency")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    
    pipe.vae.enable_tiling()
    if args.lora_checkpoint_dir:
        print(f"Loading LoRA weights from {args.lora_checkpoint_dir}")
        config_path = os.path.join(args.lora_checkpoint_dir, "lora_config.json")
        with open(config_path, "r") as f:
            lora_config_dict = json.load(f)
        rank = lora_config_dict["lora_params"]["lora_rank"]
        lora_alpha = lora_config_dict["lora_params"]["lora_alpha"]
        lora_scaling = lora_alpha / rank
        pipe.load_lora_weights(args.lora_checkpoint_dir, adapter_name="default")
        pipe.set_adapters(["default"], [lora_scaling])

    # Set default prompts if not provided
    if args.prompt is None:
        print(colored("Using default prompt", "red"))
        args.prompt = "A cat walks on the grass, realistic"
    print("=" * 20 + " Prompts " + "=" * 20)
    print(f"Prompt: {args.prompt}\n\n" + f"Negative Prompt: {args.negative_prompt}")
    if args.negative_prompt is None:
        args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    # here we replace the attentinon processor with our customized attention api based on flashinfer
    replace_hunyuan_attention(
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
        
    # Generate video
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
    ).frames[0]
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    export_to_video(output, args.output_file, fps=24, backend="imageio")
