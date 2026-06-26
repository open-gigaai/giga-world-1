import os
import sys
import json
sys.path.append("/mnt/pfs/users/zhanqian.wu/code/Gigaworld")

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import time
import cv2
from PIL import Image

import pandas as pd
import torch
from tqdm import tqdm


from gigaworld.diffusers_version.scheduling_gigaworld_diffusers import GigaworldScheduler

from gigaworld.diffusers_version.pipeline_gigaworld_diffusers import GigaworldFunCtrlPipeline
from gigaworld.diffusers_version.transformer_gigaworld_functrl_diffusers import GigaworldTransformer3DModelFunCtrl

# from gigaworld.modules.transformer_functrl_gigaworld import GigaworldTransformer3DModelFunCtrl
# from gigaworld.pipelines.pipeline_gigaworld_functrl import GigaworldFunCtrlPipeline

from gigaworld.modules.gigaworld_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from gigaworld.utils.utils_base import load_extra_components

from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image, load_video


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video with model")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Gigaworld-Base")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="BestWishYsh/Gigaworld-Base",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--partial_path",
        type=str,
        default=None,
    )
    parser.add_argument("--output_folder", type=str, default="./output_gigaworld")
    parser.add_argument("--enable_compile", action="store_true")

    # === Generation parameters ===
    parser.add_argument(
        "--sample_type",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "v2v"],
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=99)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)
    parser.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--is_skip_first_chunk", action="store_true")
    parser.add_argument("--is_amplify_first_chunk", action="store_true")

    # === Prompts ===
    parser.add_argument("--use_interpolate_prompt", action="store_true")
    parser.add_argument("--interpolation_steps", type=int, default=3)
    parser.add_argument("--interpolate_time", type=int, default=7)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--image_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--image_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--video_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--control_video_path", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. Medium shot focusing on the train window and the rushing scenery beyond.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )
    parser.add_argument("--prompt_txt_path", type=str, default=None)
    parser.add_argument("--base_image_prompt_path", type=str, default=None)
    parser.add_argument("--image_prompt_csv_path", type=str, default=None)
    parser.add_argument("--interactive_prompt_csv_path", type=str, default=None)

    # === Group-Offloading ===
    parser.add_argument("--enable_low_vram_mode", action="store_true")
    parser.add_argument(
        "--group_offloading_type",
        type=str,
        choices=["leaf_level", "block_level"],
        default="leaf_level",
    )
    parser.add_argument("--num_blocks_per_group", type=str, default="4")

    return parser.parse_args()

def get_first_frame_image(video_path, width, height):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).resize((width, height))
    return img

def materialize_norm_out_if_meta(
    transformer,
    dtype=torch.bfloat16,
):
    p = transformer.norm_out.scale_shift_table

    if not p.is_meta:
        print("✅ norm_out.scale_shift_table already materialized")
        return

    print(
        f"⚠️ materialize norm_out.scale_shift_table: "
        f"shape={tuple(p.shape)}, dtype={dtype}"
    )

    transformer.norm_out.scale_shift_table = torch.nn.Parameter(
        torch.zeros(
            tuple(p.shape),
            dtype=dtype,
            device="cpu",
        ),
        requires_grad=p.requires_grad,
    )

def main():
    args = parse_args()

    assert not (args.enable_low_vram_mode and args.enable_compile), (
        "enable_low_vram_mode and enable_compile cannot be used together."
    )

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    os.makedirs(args.output_folder, exist_ok=True)

    # 纯单卡设备设置（无任何并行）
    rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = args.prompt
    image_path = args.image_path if args.image_path is not None else None   
    video_path = args.video_path
    control_video_path = args.control_video_path
    interpolate_time_list = None

    # 加载模型
    # transformer = GigaworldTransformer3DModelFunCtrl.from_pretrained(
    #     args.transformer_path,
    #     subfolder="transformer",
    #     torch_dtype=args.weight_dtype,
    # )

    transformer = GigaworldTransformer3DModelFunCtrl.from_pretrained(
        "/mnt/pfs/users/zhanqian.wu/code/Gigaworld/ablation_stage_1_post_giga_functrl_lora_0522_debug/checkpoint-2/",
        subfolder="transformer_full",
        torch_dtype=args.weight_dtype,
    )
    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()

    # 自动设置最优注意力加速
    # cuda_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    # if cuda_major >= 9:
    #     try:
    #         transformer.set_attention_backend("_flash_3_hub")
    #     except Exception:
    #         transformer.set_attention_backend("flash_hub")
    # else:
    #     transformer.set_attention_backend("flash_hub")

    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    scheduler = GigaworldScheduler.from_pretrained(
        args.base_model_path,
        subfolder="scheduler",
    )
    pipe = GigaworldFunCtrlPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=args.weight_dtype,
    )

    # # # LoRA 加载
    # if args.lora_path is not None:
    #     pipe.load_lora_weights(args.lora_path, adapter_name="default")
    #     pipe.set_adapters(["default"], adapter_weights=[1.0])
    #     if args.partial_path is not None:
    #         from argparse import Namespace
    #         args.training_config = Namespace()
    #         args.training_config.is_enable_stage1 = True
    #         args.training_config.restrict_self_attn = True
    #         args.training_config.is_amplify_history = True
    #         args.training_config.is_use_gan = True
    #         load_extra_components(args, transformer, args.partial_path)

    # 编译加速
    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    # 低显存 / 正常加载
    if args.enable_low_vram_mode:
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type=args.group_offloading_type,
            num_blocks_per_group=args.num_blocks_per_group if args.group_offloading_type == "block_level" else None,
            use_stream=True,
            record_stream=True,
        )
    else:
        # materialize_norm_out_if_meta(
        #     pipe.transformer,
        #     dtype=args.weight_dtype,
        # )
        import ipdb; ipdb.set_trace()

        # ====================== 查看 meta 张量 ======================
        print("\n" + "="*50)
        print("🔍 检查模型中所有 meta 张量：")
        for name, p in pipe.transformer.named_parameters():
            if p.is_meta:
                print(name)
        print("="*50 + "\n")
        # ===========================================================
        pipe = pipe.to(device)
    
    
    # ==================== 生成视频逻辑 ====================
    if args.prompt_txt_path is not None:
        with open(args.prompt_txt_path, "r") as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]

        for idx, prompt in tqdm(enumerate(prompt_list), desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{idx}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device=device.type).manual_seed(args.seed),
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        image=load_image(image_path).resize((args.width, args.height)) if image_path else None,
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        video=load_video(video_path) if video_path else None,
                        video_noise_sigma_min=args.video_noise_sigma_min,
                        video_noise_sigma_max=args.video_noise_sigma_max,
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                    ).frames[0]
                except Exception:
                    continue

            export_to_video(output, output_path, fps=10)

    elif args.image_prompt_csv_path is not None:
        df = pd.read_csv(args.image_prompt_csv_path)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{row['id']}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            prompt = row.get("refined_prompt") or row["prompt"]
            image_path = os.path.join(args.base_image_prompt_path, row["image_name"])

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device=device.type).manual_seed(args.seed),
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        image=load_image(image_path).resize((args.width, args.height)),
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        video=None,
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                    ).frames[0]
                except Exception:
                    continue

            export_to_video(output, output_path, fps=10)

    elif args.interactive_prompt_csv_path is not None:
        df = pd.read_csv(args.interactive_prompt_csv_path)
        df = df.sort_values(by=["id", "prompt_index"])
        all_video_ids = df["id"].unique()

        for video_id in tqdm(all_video_ids, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{video_id}.mp4")
            if os.path.exists(output_path):
                print(f"skipping {output_path}!")
                continue

            group_df = df[df["id"] == video_id]
            prompt_list = group_df["refined_prompt"].fillna(group_df["prompt"]).tolist() if "refined_prompt" in df.columns else group_df["prompt"].tolist()
            interpolate_time_list = [args.interpolate_time] * len(prompt_list)

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt_list,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device=device.type).manual_seed(args.seed),
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        image=load_image(image_path).resize((args.width, args.height)) if image_path else None,
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        video=load_video(video_path) if video_path else None,
                        video_noise_sigma_min=args.video_noise_sigma_min,
                        video_noise_sigma_max=args.video_noise_sigma_max,
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                    ).frames[0]
                except Exception:
                    continue

            export_to_video(output, output_path, fps=10)

    else:
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=device.type).manual_seed(args.seed),
                history_sizes=[16, 2, 1],
                # num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                # keep_first_frame=True,
                # is_skip_first_chunk=args.is_skip_first_chunk,
                is_amplify_first_chunk=args.is_amplify_first_chunk,
                use_zero_init=args.use_zero_init,
                zero_steps=args.zero_steps,
                image=load_image(image_path).resize((args.width, args.height)) if image_path else None,
                image_noise_sigma_min=args.image_noise_sigma_min,
                image_noise_sigma_max=args.image_noise_sigma_max,
                control_video=load_video(control_video_path) if control_video_path else None,   
                #video=load_video(video_path) if video_path else None,
                video_noise_sigma_min=args.video_noise_sigma_min,
                video_noise_sigma_max=args.video_noise_sigma_max,
                use_interpolate_prompt=args.use_interpolate_prompt,
                interpolation_steps=args.interpolation_steps,
                interpolate_time_list=interpolate_time_list,
                add_noise_to_image_latents=False,
            ).frames[0]

        file_count = len([f for f in os.listdir(args.output_folder) if os.path.isfile(os.path.join(args.output_folder, f))])
        save_prefix = f"{file_count:04d}_{args.sample_type}_{int(time.time())}"
        output_path = os.path.join(
            args.output_folder,
            f"{save_prefix}_gen.mp4"
        )
        export_to_video(output, output_path, fps=10)
        import ipdb; ipdb.set_trace()
        # ------------------------------------------------
        # 2. save gt
        # ------------------------------------------------
        if video_path is not None and os.path.isfile(video_path):
            gt_video = load_video(video_path)
            gt_output_path = os.path.join(args.output_folder,f"{save_prefix}_gt.mp4")
            export_to_video(gt_video,gt_output_path,fps=10,)
            print(f"✅ GT saved: {gt_output_path}")

        # ------------------------------------------------
        # 3. save control
        # ------------------------------------------------
        if control_video_path is not None and os.path.isfile(control_video_path):
            ctrl_video = load_video(control_video_path)
            ctrl_output_path = os.path.join(args.output_folder,f"{save_prefix}_control.mp4")
            export_to_video(ctrl_video,ctrl_output_path,fps=10,)
            print(f"✅ Control saved: {ctrl_output_path}")
        
        if image_path is not None and os.path.isfile(image_path):
            first_frame_save_path = os.path.join(args.output_folder,f"{save_prefix}_first_frame.jpg")
            Image.open(image_path).save(first_frame_save_path)
            print(f"✅ First frame saved: {first_frame_save_path}")
        
        print(f"prompt: {prompt}")


    if torch.cuda.is_available():
        print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")


if __name__ == "__main__":
    import sys
    sys.argv = [
        sys.argv[0],
        # === Model paths ===
        "--base_model_path", "/mnt/pfs/users/zhanqian.wu/ckpt/Wan2.1-Fun-V1.1-1.3B-giga-ctrl-2200",
        "--transformer_path", "/mnt/pfs/users/zhanqian.wu/ckpt/Wan2.1-Fun-V1.1-1.3B-giga-ctrl-2200",
        "--lora_path", "/mnt/pfs/users/zhanqian.wu/code/Gigaworld/ablation_stage_1_post_giga_functrl_lora_0522/checkpoint-3200/pytorch_lora_weights.safetensors",
        "--partial_path", "/mnt/pfs/users/zhanqian.wu/code/Gigaworld/ablation_stage_1_post_giga_functrl_lora_0522/checkpoint-3200/transformer_partial.pth",
        "--output_folder", "./output_gigaworld",
        "--enable_compile",

        # === Generation parameters ===
        "--sample_type", "i2v",
        "--weight_dtype", "bf16",
        "--seed", "42",
        "--height", "480",
        "--width", "1920",
        "--num_frames", "99",
        "--fps", "10",
        "--num_inference_steps", "15",
        "--guidance_scale", "5.0",
        "--use_zero_init",
        "--image_noise_sigma_min", "0.111",
        "--image_noise_sigma_max", "0.135",

        # === Prompts ===
        "--image_path", None,  # 👈 你的图片路径
        "--prompt", None,   
        "--control_video_path", None,
        "--negative_prompt", "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    ]

    # 定义路径
    base_folder = "/shared_disk/users/zhanqian.wu/data/infer_data/helios_data/giga_ctrl_3view/task11"
    video_folder = f"{base_folder}/videos"
    control_video_folder = f"{base_folder}/control_videos"
    json_path = f"{base_folder}/helios_giga_ctrl.jsonl"

    # 读取 JSON 文件获取前50个视频对
    video_pairs = []
    max_pairs = 50
    
    if os.path.exists(json_path):

        # ------------------------------------------------
        # jsonl
        # ------------------------------------------------
        if json_path.endswith(".jsonl"):

            data = []

            with open(json_path, "r") as f:
                for line in f:
                    line = line.strip()

                    if len(line) == 0:
                        continue

                    data.append(json.loads(line))

        # ------------------------------------------------
        # json
        # ------------------------------------------------
        else:
            with open(json_path, "r") as f:
                data = json.load(f)

            
        # 只取前50个
        data = data[:max_pairs]
        
        print("📹 从 JSON 获取前50个视频对：")
        print("=" * 100)
        for idx, item in enumerate(data, 1):
            # 获取视频路径
            video_rel_path = item.get("path", "")
            control_rel_path = item.get("control_path", "")
            
            # 处理 caption（cap 是数组）
            cap_list = item.get("cap", [])
            caption = cap_list[0] if cap_list else ""
            
            if video_rel_path:
                full_video_path = os.path.join(video_folder, video_rel_path)
                full_control_path = os.path.join(control_video_folder, control_rel_path)
                
                # 检查文件是否存在
                video_exists = os.path.exists(full_video_path)
                control_exists = os.path.exists(full_control_path)
                
                video_pairs.append({
                    "video_path": full_video_path,
                    "control_video_path": full_control_path,
                    "caption": caption,
                    "num_frames": item.get("num_frames", 0),
                    "fps": item.get("fps", 0),
                    "resolution": item.get("resolution", {}),
                    "video_exists": video_exists,
                    "control_exists": control_exists
                })
                
                status = "✅" if video_exists and control_exists else "⚠️"
                print(f"{status} {idx:3d}. Video: {video_rel_path}")
                print(f"    Control: {control_rel_path}")
                print(f"    Frames: {item.get('num_frames', 0)}, FPS: {item.get('fps', 0)}")
                print(f"    Caption: {caption[:50]}..." if len(caption) > 50 else f"    Caption: {caption}")
                if not video_exists:
                    print(f"    ⚠️ Video file missing")
                if not control_exists:
                    print(f"    ⚠️ Control file missing")
                print()
        
        print("=" * 100)
        print(f"共找到 {len(video_pairs)} 个视频对（最多50个）")
        
    else:
        print(f"❌ JSON 文件不存在: {json_path}")
    
    if video_pairs:
        first_pair = video_pairs[-1]

        img = get_first_frame_image(first_pair["video_path"], 1920, 480)
        temp_img = "./temp_gt_first_frame.jpg"
        img.save(temp_img)
        
        for i in range(len(sys.argv)):
            if sys.argv[i] == "--control_video_path" and i+1 < len(sys.argv):
                sys.argv[i+1] = first_pair["control_video_path"]
            if sys.argv[i] == "--prompt" and i+1 < len(sys.argv):
                sys.argv[i+1] = first_pair["caption"]
            if sys.argv[i] == "--image_path":
                sys.argv[i+1] = temp_img
        
        print(f"\n🎯 已设置推理输入:")
        print(f"   video_path: {first_pair['video_path']}")
        print(f"   control_video_path: {first_pair['control_video_path']}")
        print(f"   prompt: {first_pair['caption'][:100]}...")

    main()