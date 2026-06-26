#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ============================================================
# ENV
# ============================================================
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

import argparse
import cv2
from PIL import Image

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from peft import LoraConfig, set_peft_model_state_dict
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import export_to_video, load_image, load_video, convert_unet_state_dict_to_peft

from gigaworld.modules.gigaworld_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from gigaworld.modules.transformer_functrl_gigaworld import GigaworldTransformer3DModelFunCtrl
from gigaworld.pipelines.pipeline_gigaworld_functrl import GigaworldFunCtrlPipeline
from gigaworld.utils.train_config import Args
from gigaworld.utils.utils_base import (
    NORM_LAYER_PREFIXES,
    load_model_checkpoint,
    load_extra_components,
)


def get_first_frame_image(video_path, width, height):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb).resize((width, height))


def normalize_optional_path(x):
    if x is None:
        return None
    if str(x).lower() in ["none", "null", ""]:
        return None
    return x


def build_lora_config(args, transformer):
    if args.model_config.lora_layers is not None:
        if args.model_config.lora_layers != "all-linear":
            target_modules = [x.strip() for x in args.model_config.lora_layers.split(",")]

            if args.training_config.is_train_lora_patch_embedding and "patch_embedding" not in target_modules:
                target_modules.append("patch_embedding")

            if args.training_config.is_train_lora_multi_term_memory_patchg:
                for patch_name in ["patch_short", "patch_mid", "patch_long"]:
                    if patch_name not in target_modules:
                        target_modules.append(patch_name)
        else:
            target_modules = set()
            for name, module in transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)

            target_modules = list(target_modules)

            if args.training_config.is_train_lora_patch_embedding and "patch_embedding" not in target_modules:
                target_modules.append("patch_embedding")

            if args.training_config.is_train_lora_multi_term_memory_patchg:
                for patch_name in ["patch_short", "patch_mid", "patch_long"]:
                    if patch_name not in target_modules:
                        target_modules.append(patch_name)

        target_modules = [t for t in target_modules if "norm" not in t]
    else:
        target_modules = list(args.model_config.lora_target_modules)

    target_modules = [t for t in target_modules if "control_branch" not in t]

    lora_exclude_modules = list(args.model_config.lora_exclude_modules)
    if "control_branch" not in lora_exclude_modules:
        lora_exclude_modules.append("control_branch")

    return LoraConfig(
        r=args.model_config.lora_rank,
        lora_alpha=args.model_config.lora_alpha,
        lora_dropout=args.model_config.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
        exclude_modules=lora_exclude_modules,
    )


def resolve_transformer_load_path(cli_args, args):
    """
    支持两种写法：

    1. 传父目录：
       --transformer_model_name_or_path /xxx/ablation_xxx/
       且里面有 transformer/config.json
       => path=/xxx/ablation_xxx, subfolder=transformer

    2. 直接传 transformer 目录：
       --transformer_model_name_or_path /xxx/ablation_xxx/transformer
       => path=/xxx/ablation_xxx/transformer, subfolder=None
    """

    if cli_args.transformer_model_name_or_path is None:
        return cli_args.base_model_path, args.model_config.subfolder or "transformer"

    p = cli_args.transformer_model_name_or_path.rstrip("/")

    if os.path.exists(os.path.join(p, "config.json")):
        return p, None

    if os.path.exists(os.path.join(p, "transformer", "config.json")):
        return p, "transformer"

    # fallback：按用户传入路径直接加载
    return p, None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--transformer_model_name_or_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    parser.add_argument("--control_video_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_frames", type=int, default=99)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--sample_name", type=str, default="sample")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--captions_path", type=str, default=None)
    parser.add_argument("--enable_tiling", type=bool, default=False)
    cli_args = parser.parse_args()

    os.makedirs(cli_args.output_dir, exist_ok=True)
    output_video_dir = os.path.join(cli_args.output_dir, "videos")
    os.makedirs(output_video_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    checkpoint_path = normalize_optional_path(cli_args.checkpoint_path)

    config = OmegaConf.load(cli_args.config)
    schema = OmegaConf.structured(Args)
    args = OmegaConf.merge(schema, config)
    args.model_config.pretrained_model_name_or_path = cli_args.base_model_path
    args.model_config.load_checkpoints_custom = checkpoint_path is not None
    args.model_config.load_model_path = checkpoint_path

    transformer_load_path, transformer_subfolder = resolve_transformer_load_path(cli_args, args)

    args.model_config.transformer_model_name_or_path = transformer_load_path
    args.model_config.subfolder = transformer_subfolder

    set_seed(cli_args.seed)

    accelerator = Accelerator(
        mixed_precision=args.training_config.mixed_precision,
    )

    device = accelerator.device

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    accelerator.print("=" * 100)
    accelerator.print("🚀 Infer like train validation")
    accelerator.print("=" * 100)
    accelerator.print(f"device               : {device}")
    accelerator.print(f"weight_dtype         : {weight_dtype}")
    accelerator.print(f"base_model           : {cli_args.base_model_path}")
    accelerator.print(f"transformer_path     : {transformer_load_path}")
    accelerator.print(f"transformer_subfolder: {transformer_subfolder}")
    accelerator.print(f"checkpoint           : {checkpoint_path}")
    accelerator.print(f"control_video        : {cli_args.control_video_path}")
    accelerator.print(f"image                : {cli_args.image_path}")
    accelerator.print(f"output_dir           : {cli_args.output_dir}")
    accelerator.print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        cli_args.base_model_path,
        subfolder="tokenizer",
        revision=args.model_config.revision,
    )

    noise_scheduler = UniPCMultistepScheduler.from_pretrained(
        "scripts/accelerate_configs/scheduler_config.json"
    )

    vae = AutoencoderKLWan.from_pretrained(
        cli_args.base_model_path,
        subfolder="vae",
        revision=args.model_config.revision,
        variant=args.model_config.variant,
        torch_dtype=weight_dtype,
        device_map=device,
    )

    if args.model_config.enable_slicing:
        print("[🐱 INFO] 启用 VAE slicing")
        vae.enable_slicing()

    vae.enable_tiling()

    text_encoder = UMT5EncoderModel.from_pretrained(
        cli_args.base_model_path,
        subfolder="text_encoder",
        revision=args.model_config.revision,
        variant=args.model_config.variant,
        dtype=weight_dtype,
        device_map=device,
    )

    vae.eval()
    text_encoder.eval()

    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
        "zero_history_timestep": args.training_config.zero_history_timestep,
        "restrict_self_attn": args.training_config.restrict_self_attn,
        "guidance_cross_attn": args.training_config.guidance_cross_attn,
        "is_train_restrict_lora": args.training_config.is_train_restrict_lora,
        "restrict_lora": args.training_config.restrict_lora,
        "restrict_lora_rank": args.training_config.restrict_lora_rank,
        "is_amplify_history": args.training_config.is_amplify_history,
        "history_scale_mode": args.training_config.history_scale_mode,
        "model_type": args.model_config.model_type,
    }

    accelerator.print("🏗️ Loading transformer...")
    accelerator.print(f"   path     : {transformer_load_path}")
    accelerator.print(f"   subfolder: {transformer_subfolder}")

    transformer = GigaworldTransformer3DModelFunCtrl.from_pretrained(
        transformer_load_path,
        subfolder=transformer_subfolder,
        transformer_additional_kwargs=transformer_additional_kwargs,
    )

    accelerator.print("✅ Transformer loaded.")

    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()

    transformer.requires_grad_(False)

    if checkpoint_path is not None:
        transformer_lora_config = build_lora_config(args, transformer)
        transformer.add_adapter(transformer_lora_config)
        accelerator.print("🔁 Loading LoRA checkpoint with train load_model_checkpoint()...")
        load_model_checkpoint(
            args=args,
            checkpoint_path=checkpoint_path,
            transformer=transformer,
            pipeline_class=GigaworldFunCtrlPipeline,
            norm_layer_prefixes=NORM_LAYER_PREFIXES,
            convert_unet_state_dict_to_peft_fn=convert_unet_state_dict_to_peft,
            set_peft_model_state_dict_fn=set_peft_model_state_dict,
            cast_training_params_fn=cast_training_params,
        )

        partial_path = os.path.join(
            checkpoint_path,
            "transformer_partial.pth",
        )

        if os.path.exists(partial_path):
            accelerator.print(f"🔁 Loading extra components: {partial_path}")
            load_extra_components(
                args,
                transformer,
                partial_path,
            )
        else:
            accelerator.print(f"⚠️ transformer_partial.pth not found: {partial_path}")
    else:
        accelerator.print("ℹ️ checkpoint_path is None, skip LoRA checkpoint and extra components.")

    for name, param in transformer.named_parameters():
        should_keep_fp32 = any(
            pattern in name
            for pattern in transformer.__class__._keep_in_fp32_modules
        )

        if should_keep_fp32:
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(weight_dtype)

    transformer.to(device)
    transformer.eval() 

    meta_names = [name for name, p in transformer.named_parameters() if p.is_meta]

    if len(meta_names) > 0:
        accelerator.print("❌ Found meta parameters:")
        for name in meta_names:
            accelerator.print(f"   {name}")
        raise RuntimeError("Meta parameters still exist after loading checkpoint.")

    pipe = GigaworldFunCtrlPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=noise_scheduler,
        pipeline_type=args.model_config.model_type,
    )

    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(cli_args.seed)

    base_folder = cli_args.dataset_dir
    json_path = cli_args.captions_path
    if base_folder is None and json_path is not None:
        base_folder = os.path.dirname(json_path)
    if json_path is None and base_folder is not None:
        json_path = os.path.join(base_folder, "captions.json")

    video_folder = os.path.join(base_folder, "videos") if base_folder is not None else None
    control_video_folder = os.path.join(base_folder, "control_videos") if base_folder is not None else None

    video_pairs = []
    if json_path is not None and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data, 1):
            video_rel_path = item.get("path", "")
            control_rel_path = item.get("control_path", "")
            prompt = (item.get("cap") or [""])[0]
            num_frames = item.get("num_frames", 0)

            if num_frames < 99 or not video_rel_path or not control_rel_path or not prompt:
                continue

            assert video_folder is not None
            assert control_video_folder is not None
            full_video_path = os.path.join(video_folder, os.path.basename(video_rel_path))
            full_control_path = os.path.join(control_video_folder, os.path.basename(control_rel_path))
            video_exists = os.path.exists(full_video_path)
            control_exists = os.path.exists(full_control_path)

            status = "✅" if video_exists and control_exists else "⚠️"
            print(f"{status} {idx:3d}. Video: {video_rel_path}")
            print(f"    Control: {control_rel_path}")
            print(f"    Frames: {num_frames}, FPS: {item.get('fps', 0)}")

            if not video_exists:
                print("    ⚠️ Video file missing")
            if not control_exists:
                print("    ⚠️ Control file missing")
            if not video_exists or not control_exists:
                continue

            video_pairs.append(
                {
                    "video_path": full_video_path,
                    "control_video_path": full_control_path,
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "fps": item.get("fps", cli_args.fps),
                    "sample_name": os.path.splitext(os.path.basename(video_rel_path))[0],
                }
            )

        print("=" * 100)
        print(f"共找到 {len(video_pairs)} 个可推理视频对")
    else:
        control_video_path = normalize_optional_path(cli_args.control_video_path)
        image_path = normalize_optional_path(cli_args.image_path)
        prompt = normalize_optional_path(cli_args.prompt)

        assert control_video_path is not None
        assert image_path is not None
        assert prompt is not None
        assert os.path.isfile(control_video_path), f"control video not found: {control_video_path}"
        assert os.path.isfile(image_path), f"image not found: {image_path}"

        video_pairs.append(
            {
                "control_video_path": control_video_path,
                "image_path": image_path,
                "prompt": prompt,
                "num_frames": cli_args.num_frames,
                "fps": cli_args.fps,
                "sample_name": cli_args.sample_name,
            }
        )

    for idx, pair in enumerate(video_pairs):
        accelerator.print(f"\n🚀 开始推理第 {idx + 1}/{len(video_pairs)} 个视频...")

        control_video = load_video(pair["control_video_path"])
        if "image_path" in pair:
            image = load_image(pair["image_path"]).resize((cli_args.width, cli_args.height))
        else:
            image = get_first_frame_image(pair["video_path"], cli_args.width, cli_args.height)
            if image is None:
                accelerator.print(f"⚠️ 读取首帧失败，跳过: {pair['video_path']}")
                continue

        pipeline_args = {
            "prompt": pair["prompt"],
            "negative_prompt": args.data_config.negative_prompt,
            "guidance_scale": cli_args.guidance_scale,
            "num_frames": cli_args.num_frames,
            "height": cli_args.height,
            "width": cli_args.width,
            "num_inference_steps": cli_args.num_inference_steps,
            "use_dynamic_shifting": args.validation_config.use_dynamic_shifting,
            "time_shift_type": args.validation_config.time_shift_type,
            "history_sizes": args.training_config.history_sizes,
            "latent_window_size": args.validation_config.validation_latent_window_size[0],
            "is_keep_x0": True,
            "use_kv_cache": args.validation_config.use_kv_cache,
            "use_dmd": False,
            "control_video": control_video,
            "image": image,
        }

        accelerator.print("🎬 Start inference...")
        accelerator.print(f"📝 prompt: {pair['prompt']}")
        accelerator.print(
            f"📐 {cli_args.width}x{cli_args.height}, "
            f"frames={cli_args.num_frames}, "
            f"steps={cli_args.num_inference_steps}, "
            f"cfg={cli_args.guidance_scale}"
        )

        output = pipe(
            **pipeline_args,
            generator=generator,
            output_type="np",
        ).frames[0]

        gen_path = os.path.join(output_video_dir, f"{pair['sample_name']}.mp4")

        export_to_video(output, gen_path, fps=cli_args.fps)

        accelerator.print(f"✅ saved gen       : {gen_path}")

        del control_video
        del image
        del output
        free_memory()

    del pipe
    del transformer
    del vae
    del text_encoder
    free_memory()

    accelerator.print("🎉 Done.")


if __name__ == "__main__":
    main()

