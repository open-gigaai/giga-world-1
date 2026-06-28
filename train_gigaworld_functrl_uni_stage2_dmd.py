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

import os

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"
from omegaconf import OmegaConf
import argparse
import copy
import json
import logging
import math
import random
import shutil
import threading
from datetime import timedelta
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    broadcast,
    set_seed,
)
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _collate_lora_metadata, cast_training_params, free_memory
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

import diffusers
from gigaworld.dataset.dataloader_dmd import BucketedFeatureDataset, BucketedSampler, collate_fn
from gigaworld.modules.gigaworld_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)

from gigaworld.modules.transformer_functrl_gigaworld import GigaworldTransformer3DModelFunCtrl
from gigaworld.pipelines.pipeline_gigaworld_functrl import GigaworldFunCtrlPipeline
from gigaworld.scheduler.scheduling_gigaworld import GigaworldScheduler
from gigaworld.utils.create_ema_zero3_lora import create_ema_final, gather_zero3ema
from gigaworld.utils.train_config import Args
from gigaworld.utils.utils_base import (
    NORM_LAYER_PREFIXES,
    compare_configs,
    encode_prompt,
    get_optimizer,
    load_extra_components,
    load_model_checkpoint,
    save_extra_components,
    save_model_checkpoint,
)
from gigaworld.utils.utils_gigaworld_post import (
    OptimizedLowVRAMManager,
    _critic_loss,
    _generator_loss,
    merge_dict_list,
    sample_dynamic_dmd_num_latent_sections,
)
from diffusers.utils import export_to_video, load_image, load_video

if is_wandb_available():
    import wandb

check_min_version("0.36.0.dev0")
logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def build_target_modules(transformer, args):
    if args.model_config.lora_layers is not None:
        if args.model_config.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in args.model_config.lora_layers.split(",")]
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
        target_modules = args.model_config.lora_target_modules
    
    print_lora_target_modules(args, Accelerator, target_modules)
    return target_modules

def apply_extra_trainable_modules(transformer, args):
    trainable_modules = []
    if args.training_config.is_train_full_multi_term_memory_patchg:
        trainable_modules.extend(["patch_short", "patch_mid", "patch_long"])
    if args.training_config.is_train_full_patch_embedding:
        trainable_modules.append("patch_embedding")
    if args.training_config.is_train_restrict_lora:
        trainable_modules.extend(["q_loras", "k_loras", "v_loras"])
    if args.training_config.is_amplify_history:
        trainable_modules.append("history_key_scale")

    for name, param in transformer.named_parameters():
        for module_name in trainable_modules:
            if module_name in name:
                param.requires_grad = True
                break
    
def safe_item(value):
    return value.item() if hasattr(value, "item") else value


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


def rank0_print(accelerator, *args, **kwargs):
    if accelerator.is_main_process:
        print(*args, **kwargs)

def print_model_info(accelerator, name, model):
    if not accelerator.is_main_process:
        return
    try:
        first_param = next(model.parameters())
        dtype = first_param.dtype
        device = first_param.device
    except StopIteration:
        dtype = "N/A"
        device = "N/A"
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(
        accelerator,
        f"\033[96m[{name}]\033[0m "
        f"\033[90mclass={model.__class__.__name__} "
        f"dtype={dtype} device={device}\033[0m "
        f"\033[92mtotal={total_params / 1e9:.3f}B\033[0m "
        f"\033[93mtrainable={trainable_params / 1e6:.3f}M\033[0m",
    )

def print_lora_target_modules(args, accelerator, target_modules):
    """
    Print a compact summary of LoRA target_modules configuration.
    """
    n_patch_embedding = sum(1 for t in target_modules if "patch_embedding" in t)
    n_patch = sum(1 for t in target_modules if any(p in t for p in ["patch_short", "patch_mid", "patch_long"]))
    n_norm = sum(1 for t in target_modules if "norm" in t.lower())
    n_linear = len(target_modules) - n_patch_embedding - n_patch - n_norm

    rank0_print(
        accelerator,
        f"\033[96m[LoRA]\033[0m "
        f"\033[92mtarget_modules={len(target_modules)}\033[0m "
        f"\033[94mlinear={n_linear} patch_embed={n_patch_embedding} patch={n_patch} norm_excluded={n_norm}\033[0m",
    )

def print_trainable_parameters(model, accelerator, title="Trainable Parameters"):
    """
    Print a compact summary of model trainable parameters.
    """
    total_params = 0
    trainable_params = 0
    trainable_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)

    trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0

    if not trainable_names:
        rank0_print(accelerator, f"\033[91m[NoTrainable] {title}: 0 parameters!\033[0m")
        return

    # Classify by name pattern (avoid materializing sorted list)
    n_lora = sum(1 for n in trainable_names if "lora" in n.lower())
    n_patch = sum(1 for n in trainable_names if any(p in n for p in ["patch_", "short", "mid", "long"]))
    n_norm = sum(1 for n in trainable_names if "norm" in n.lower())
    n_scale = sum(1 for n in trainable_names if "scale" in n.lower())

    summary = (
        f"\033[96m[{title}]\033[0m "
        f"\033[92mtrainable={trainable_params / 1e6:.3f}M\033[0m "
        f"({trainable_percent:.4f}%) "
        f"\033[90mfrozen={(total_params - trainable_params) / 1e6:.3f}M\033[0m "
        f"\033[94mlora={n_lora} patch={n_patch} norm={n_norm} scale={n_scale}\033[0m"
    )
    rank0_print(accelerator, summary)

def print_dataset_info(accelerator, dataset_kwargs, train_dataset, train_dataloader, sampler):
    if not accelerator.is_main_process:
        return
    rank0_print(
        accelerator,
        f"\033[96m[Dataset]\033[0m "
        f"\033[94mtext={dataset_kwargs.get('text_folders')} "
        f"gan={dataset_kwargs.get('gan_folders')} "
        f"ode={dataset_kwargs.get('ode_folders')}\033[0m "
        f"\033[90mshape={dataset_kwargs.get('single_res')} "
        f"frames={dataset_kwargs.get('single_num_frame')} "
        f"len={dataset_kwargs.get('single_length')}\033[0m",
    )
    rank0_print(
        accelerator,
        f"\033[96m[Dataset]\033[0m "
        f"\033[92mexamples={len(train_dataset):,} "
        f"batches={len(train_dataloader):,}\033[0m",
    )
    if hasattr(sampler, 'buckets'):
        rank0_print(accelerator, f"Num buckets    : {len(sampler.buckets):,}")
    rank0_print(accelerator, "=" * 80 + "\n")

def print_dmd_trainable_params(model, name, lr, accelerator):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(
        accelerator,
        f"\033[93m[{name}]\033[0m "
        f"\033[92mtrainable={n_params / 1e6:.3f}M\033[0m "
        f"\033[94mlr={lr:.2e}\033[0m",
    )

@torch.no_grad()
def run_validation_functrl(
    args,
    accelerator,
    transformer,
    tokenizer,
    vae,
    text_encoder,
    noise_scheduler,
    weight_dtype,
    global_step,
):
    if accelerator.is_main_process is False:
        return

    if args.validation_config.validation_prompts is None:
        return

    accelerator.print("🧪 Running CONTROL validation...")

    pipe = GigaworldFunCtrlPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=noise_scheduler,
        pipeline_type=args.model_config.model_type,
    )

    pipe = pipe.to(accelerator.device)

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_infer_dir = os.path.join(args.output_dir, "run_infer")
    os.makedirs(run_infer_dir, exist_ok=True)

    all_videos = []
    all_prompts = []

    def get_first_frame_image(video_path, width, height):
        ext = os.path.splitext(video_path)[-1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            img = Image.open(video_path).convert("RGB").resize((width, height))
            return img
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        if not success:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((width, height))
        return img

    temp_control_video = None
    
    if args.validation_config.validation_first_image is not None:
        img = get_first_frame_image(
            args.validation_config.validation_first_image[0],
            args.validation_config.validation_width,
            args.validation_config.validation_height,
        )
        temp_img = os.path.join(run_infer_dir, "temp_gt_first_frame_0.jpg")
        img.save(temp_img)
    else:
        temp_img = None

    control_video_path = (
        temp_control_video
        if temp_control_video is not None
        else args.validation_config.validation_control_video[0]
    )
    
    pipeline_args = {
        "negative_prompt": args.data_config.negative_prompt,
        "guidance_scale": args.validation_config.validation_guidance_scale,
        "num_frames": args.validation_config.validation_max_num_frames,
        "height": args.validation_config.validation_height,
        "width": args.validation_config.validation_width,
        "num_inference_steps": args.validation_config.num_inference_steps,
        "use_dynamic_shifting": args.validation_config.use_dynamic_shifting,
        "time_shift_type": args.validation_config.time_shift_type,
        "history_sizes": args.training_config.history_sizes,
        "latent_window_size": args.validation_config.validation_latent_window_size[0],
        "is_keep_x0": True,
        "use_kv_cache": args.validation_config.use_kv_cache,
        "use_dmd": False,
        "control_video": load_video(control_video_path),
        "image": load_image(temp_img) if temp_img is not None else None,
        "prompt": args.validation_config.validation_prompts[0],
    }

    for sample_idx in range(args.validation_config.num_validation_videos):
        accelerator.print(
            f"🎬 CONTROL validation sample {sample_idx}: prompt={pipeline_args['prompt']}"
        )

        gen_video_ori = pipe(
            **pipeline_args,
            generator=generator,
            output_type="np",
        ).frames[0]

        all_videos.append(gen_video_ori)
        all_prompts.append(pipeline_args['prompt'])

    saved_files = []
    for i, (video, prompt) in enumerate(zip(all_videos, all_prompts)):
        safe_prompt = prompt[:25].replace(" ", "_").replace("/", "_")

        gen_filename = os.path.join(
            run_infer_dir,
            f"global_step{global_step}_control_gt_gen_{i}_{safe_prompt}.mp4",
        )

        export_to_video(video, gen_filename, fps=10)
        saved_files.append(gen_filename)

        accelerator.print(f"✅ Saved validation video: {gen_filename}")

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            video_logs = [
                wandb.Video(
                    filename,
                    caption=(
                        f"{i}: generated "
                        f"| prompt={all_prompts[i]} "
                        f"| step={global_step}"
                    ),
                    format="mp4",
                )
                for i, filename in enumerate(saved_files)
            ]
            tracker.log(
                {"validation_videos": video_logs},
                step=global_step,
            )

    del pipe
    free_memory()
    vae.to("cpu", non_blocking=True)
    text_encoder.to("cpu", non_blocking=True)
    free_memory()


def make_functrl_input(noise_or_latents, control_latents, image_latents=None):
    """
    Fun-Control input:
        [noise/video, control, image_dummy]
        16 + 16 + 16 = 48
    """
    if isinstance(noise_or_latents, list):
        return [
            make_functrl_input(x, control_latents, image_latents)
            for x in noise_or_latents
        ]

    if image_latents is None:
        image_latents = torch.zeros_like(noise_or_latents)

    assert noise_or_latents.shape[1] == 16
    assert control_latents.shape[1] == 16
    assert image_latents.shape[1] == 16

    if control_latents.shape[-3:] != noise_or_latents.shape[-3:]:
        control_latents = control_latents[:, :, : noise_or_latents.shape[2]]

    return torch.cat(
        [
            noise_or_latents,
            control_latents,
            image_latents,
        ],
        dim=1,
    )

def main(args):
    if not args.training_config.is_train_dmd:
        raise ValueError("This script is DMD-only. Please set training_config.is_train_dmd=True.")
    if not args.data_config.use_stage2_dataset:
        raise ValueError("DMD-only script expects data_config.use_stage2_dataset=True.")
    if args.training_config.is_use_gan:
        raise ValueError("This DMD-only script intentionally disables GAN. Set training_config.is_use_gan=False.")
    if args.training_config.is_use_reward_model:
        raise ValueError("This DMD-only script disables reward model. Set training_config.is_use_reward_model=False.")

    if torch.backends.mps.is_available() and args.training_config.mixed_precision == "bf16":
        raise ValueError("MPS does not support bf16 training.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))

    dmd_deepspeed_training = (
        args.training_config.dmd_generator_deepspeed_config is not None
        and args.training_config.dmd_critic_deepspeed_config is not None
    )
    deepspeed_plugins = None
    if dmd_deepspeed_training:
        generator_zero_plugin = DeepSpeedPlugin(hf_ds_config=args.training_config.dmd_generator_deepspeed_config)
        critic_zero_plugin = DeepSpeedPlugin(hf_ds_config=args.training_config.dmd_critic_deepspeed_config)
        deepspeed_plugins = {"generator": generator_zero_plugin, "critic_model": critic_zero_plugin}

    accelerator = Accelerator(
        gradient_accumulation_steps=args.training_config.gradient_accumulation_steps,
        mixed_precision=args.training_config.mixed_precision,
        log_with=args.report_to.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugins=deepspeed_plugins,
        kwargs_handlers=[kwargs, init_kwargs],
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED and not dmd_deepspeed_training:
        raise ValueError("DMD training with DeepSpeed requires dmd_generator_deepspeed_config and dmd_critic_deepspeed_config.")

    critic_accelerator = Accelerator()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        
        from omegaconf import OmegaConf
        current_conf = OmegaConf.to_container(args, resolve=True)

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                existing_conf = json.load(f)

            ignore_keys = {"training_config.local_rank"}
            mismatches = compare_configs(existing_conf, current_conf, ignore_keys=ignore_keys)

            if mismatches:
                print("\n⚠️  WARNING: Config mismatches found (program will continue running):")
                for mismatch in mismatches:
                    print(f"  - {mismatch}")
                print("ℹ️  You can ignore this warning if the changes are intended.\n")
        else:
            with open(config_path, "w") as f:
                json.dump(current_conf, f, indent=4)

    if args.training_config.use_ema:
        args.training_config.ema_zero3_port = os.environ.get("MASTER_PORT", "12345")

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb to use report_to=wandb.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    rank0_print(accelerator, accelerator, "\n🚀 ===== Pure DMD Training =====")
    rank0_print(accelerator, accelerator, f"🌍 num_processes={accelerator.num_processes}")
    rank0_print(accelerator, accelerator, f"🎮 device={accelerator.device}")
    rank0_print(accelerator, accelerator, f"🧪 mixed_precision={accelerator.mixed_precision}")
    rank0_print(accelerator, accelerator, f"📁 output_dir={args.output_dir}")
    rank0_print(accelerator, accelerator, "🚫 GAN / GT-history / reward model disabled")
    rank0_print(accelerator, accelerator, "================================\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.model_config.revision,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    noise_scheduler = UniPCMultistepScheduler.from_pretrained("scripts/accelerate_configs/scheduler_config.json")
    noise_scheduler_copy = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    noise_scheduler_copy.config.flow_shift = args.training_config.dmd_timestep_shift
    critic_noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    rank0_print(accelerator, accelerator, "\n🧭 Scheduler")
    rank0_print(accelerator, accelerator, f"├── validation scheduler : {noise_scheduler.__class__.__name__}")
    rank0_print(accelerator, accelerator, f"├── generator scheduler  : {noise_scheduler_copy.__class__.__name__}")
    rank0_print(accelerator, accelerator, f"├── critic scheduler     : {critic_noise_scheduler.__class__.__name__}")
    rank0_print(accelerator, accelerator, f"└── dmd_timestep_shift   : {args.training_config.dmd_timestep_shift}\n")

    vae = AutoencoderKLWan.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.model_config.revision,
        variant=args.model_config.variant,
        torch_dtype=torch.float32,
        device_map=accelerator.device,
    )
    if args.model_config.enable_slicing:
        vae.enable_slicing()
    if args.model_config.enable_tiling:
        vae.enable_tiling()

    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.model_config.revision,
        variant=args.model_config.variant,
        dtype=weight_dtype,
        device_map=accelerator.device,
    )

    with torch.no_grad():
        negative_prompt_embeds, _ = encode_prompt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=args.data_config.negative_prompt,
            device=accelerator.device,
        )

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
        "model_type": args.model_config.model_type, # NOTE: 0608 ADD HERE 
    }

    transformer = GigaworldTransformer3DModelFunCtrl.from_pretrained(
        args.model_config.transformer_model_name_or_path,
        subfolder=args.model_config.subfolder or "transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()

    if args.model_config.real_score_model_name_or_path is None:
        print ("⚠️  real_score_model_name_or_path is not set, using the same model as transformer for real score model (no separate critic).")
        args.model_config.real_score_model_name_or_path = args.model_config.transformer_model_name_or_path

    critic_transformer_additional_kwargs = {
        "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
        "zero_history_timestep": args.training_config.zero_history_timestep,
        "restrict_self_attn": args.training_config.restrict_self_attn,
        "guidance_cross_attn": args.training_config.guidance_cross_attn,
        "is_train_restrict_lora": args.training_config.is_train_restrict_lora,
        "restrict_lora": args.training_config.restrict_lora,
        "restrict_lora_rank": args.training_config.restrict_lora_rank,
        "is_use_gan": False,
        "is_use_gan_hooks": False,
        "is_use_gan_final": False,
        "gan_cond_map_dim": args.training_config.gan_cond_map_dim,
        "gan_hooks": [],
    }

    real_score_model = GigaworldTransformer3DModelFunCtrl.from_pretrained(
        args.model_config.real_score_model_name_or_path,
        subfolder=args.model_config.critic_subfolder or "transformer",
        transformer_additional_kwargs=critic_transformer_additional_kwargs,
    )
    real_score_model = replace_rmsnorm_with_fp32(real_score_model)
    real_score_model = replace_all_norms_with_flash_norms(real_score_model)

    transformer.requires_grad_(False)
    real_score_model.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    target_modules = build_target_modules(transformer, args)
    transformer_lora_config = LoraConfig(
        r=args.model_config.lora_rank,
        lora_alpha=args.model_config.lora_alpha,
        lora_dropout=args.model_config.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
        exclude_modules=list(args.model_config.lora_exclude_modules),
    )
    transformer.add_adapter(transformer_lora_config)

    if args.model_config.train_norm_layers:
        for name, param in transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad = True

    apply_extra_trainable_modules(transformer, args)
    print_trainable_parameters(transformer, accelerator, "Final Trainable Parameters (Transformer)")

    critic_target_modules = [
        m for m in target_modules if m not in ["patch_short", "patch_mid", "patch_long", "patch_embedding"]
    ]
    critic_exclude_modules = list(args.model_config.lora_exclude_modules) + [
        "patch_short",
        "patch_mid",
        "patch_long",
        "patch_embedding",
        "gan_heads",
        "gan_final_head",
    ]
    critic_transformer_lora_config = LoraConfig(
        r=args.model_config.critic_lora_rank,
        lora_alpha=args.model_config.critic_lora_alpha,
        lora_dropout=args.model_config.critic_lora_dropout,
        init_lora_weights="gaussian",
        target_modules=critic_target_modules,
        exclude_modules=critic_exclude_modules,
    )
    real_score_model.add_adapter(critic_transformer_lora_config)

    if args.model_config.train_norm_layers:
        for name, param in real_score_model.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad = True

    critic_trainable_modules = []

    if args.model_config.load_checkpoints_custom:
        load_model_checkpoint(
            args=args,
            checkpoint_path=args.model_config.load_model_path,
            transformer=transformer,
            pipeline_class=GigaworldFunCtrlPipeline,
            norm_layer_prefixes=NORM_LAYER_PREFIXES,
            convert_unet_state_dict_to_peft_fn=convert_unet_state_dict_to_peft,
            set_peft_model_state_dict_fn=set_peft_model_state_dict,
            cast_training_params_fn=cast_training_params,
        )
        assert args.model_config.critic_lora_name_or_path is not None
        assert args.model_config.load_dcp

    if args.model_config.critic_lora_name_or_path is not None:
        load_model_checkpoint(
            args=args,
            checkpoint_path=args.model_config.critic_lora_name_or_path,
            transformer=real_score_model,
            pipeline_class=GigaworldFunCtrlPipeline,
            norm_layer_prefixes=NORM_LAYER_PREFIXES,
            convert_unet_state_dict_to_peft_fn=convert_unet_state_dict_to_peft,
            set_peft_model_state_dict_fn=set_peft_model_state_dict,
            cast_training_params_fn=cast_training_params,
        )

    if (
        not args.training_config.is_dmd_vae_decode
        and not args.training_config.is_smoothness_loss
        and not args.training_config.is_use_reward_model
    ):
        vae = None
    text_encoder = None
    free_memory()

    for name, param in transformer.named_parameters():
        if any(pattern in name for pattern in transformer.__class__._keep_in_fp32_modules):
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(weight_dtype)
    transformer.to(accelerator.device)

    for name, param in real_score_model.named_parameters():
        if any(pattern in name for pattern in real_score_model.__class__._keep_in_fp32_modules):
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(weight_dtype)
    real_score_model.to(accelerator.device)
    free_memory()

    if args.training_config.enable_npu_flash_attention:
        if is_torch_npu_available():
            transformer.enable_npu_flash_attention()
            real_score_model.enable_npu_flash_attention()
        else:
            raise ValueError("NPU flash attention requires torch_npu.")

    if args.training_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may be unstable for training; update if needed.")
            transformer.enable_xformers_memory_efficient_attention()
            real_score_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.training_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        real_score_model.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}

            for model in models:
                model = unwrap_model(model)
                transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                if args.model_config.train_norm_layers:
                    transformer_norm_layers_to_save = {
                        f"transformer.{name}": param
                        for name, param in model.named_parameters()
                        if any(k in name for k in NORM_LAYER_PREFIXES)
                    }
                    transformer_lora_layers_to_save = {
                        **transformer_lora_layers_to_save,
                        **transformer_norm_layers_to_save,
                    }
                modules_to_save["transformer"] = model
                if weights:
                    weights.pop()

            GigaworldFunCtrlPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )
            save_extra_components(args, model=model, output_dir=output_dir)

    def load_model_hook(models, input_dir):
        transformer_ = None
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                transformer_ = unwrap_model(models.pop())
        else:
            is_critic = "critic" in input_dir
            transformer_ = GigaworldTransformer3DModelFunCtrl.from_pretrained(
                args.model_config.transformer_model_name_or_path,
                subfolder=(args.model_config.critic_subfolder if is_critic else args.model_config.subfolder) or "transformer",
                transformer_additional_kwargs=critic_transformer_additional_kwargs
                if is_critic
                else transformer_additional_kwargs,
            )
            transformer_.add_adapter(critic_transformer_lora_config if is_critic else transformer_lora_config)

        # lora_state_dict = GigaworldFunCtrlPipeline.lora_state_dict(input_dir)
        # transformer_state_dict = {
        #     k.replace("transformer.", ""): v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        # }
        # transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        # incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        
        
        # if incompatible_keys is not None and getattr(incompatible_keys, "unexpected_keys", None):
        #     logger.warning(f"Unexpected keys while loading adapter: {incompatible_keys.unexpected_keys}")

        lora_state_dict = GigaworldFunCtrlPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)

        import peft.utils.save_and_load as peft_saveload

        orig_maybe_shard_state_dict_for_tp = peft_saveload._maybe_shard_state_dict_for_tp
        peft_saveload._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None

        try:
            incompatible_keys = set_peft_model_state_dict(
                transformer_,
                transformer_state_dict,
                adapter_name="default",
            )
        finally:
            peft_saveload._maybe_shard_state_dict_for_tp = orig_maybe_shard_state_dict_for_tp

        if incompatible_keys is not None and getattr(incompatible_keys, "unexpected_keys", None):
            logger.warning(f"Unexpected keys while loading adapter: {incompatible_keys.unexpected_keys}")

        if args.model_config.train_norm_layers:
            transformer_norm_state_dict = {
                k: v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
            }
            transformer_._transformer_norm_layers = GigaworldFunCtrlPipeline._load_norm_into_transformer(
                transformer_norm_state_dict,
                transformer=transformer_,
                discard_original_layers=False,
            )

        load_extra_components(args, transformer_, os.path.join(input_dir, "transformer_partial.pth"))

        if args.training_config.mixed_precision != "fp32":
            cast_training_params([transformer_])

        dcp_dir = os.path.join(input_dir, "distributed_checkpoint")
        if "critic" not in dcp_dir and "train_dataloader" in globals():
            dcp.load({"dataloader": train_dataloader}, checkpoint_id=dcp_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    critic_accelerator.register_save_state_pre_hook(save_model_hook)
    critic_accelerator.register_load_state_pre_hook(load_model_hook)

    if args.training_config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.training_config.scale_lr:
        args.training_config.learning_rate *= (
            args.training_config.gradient_accumulation_steps
            * args.training_config.train_batch_size
            * accelerator.num_processes
        )
        args.training_config.critic_learning_rate *= (
            args.training_config.gradient_accumulation_steps
            * args.training_config.train_batch_size
            * accelerator.num_processes
        )

    if args.training_config.mixed_precision != "fp32":
        cast_training_params([transformer, real_score_model], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    critic_lora_parameters = list(filter(lambda p: p.requires_grad, real_score_model.parameters()))

    params_to_optimize = [{"params": transformer_lora_parameters, "lr": args.training_config.learning_rate}]
    critic_params_to_optimize = [{"params": critic_lora_parameters, "lr": args.training_config.critic_learning_rate}]

    print_dmd_trainable_params(transformer, "🎯Transformer", args.training_config.learning_rate, accelerator)
    print_dmd_trainable_params(real_score_model, "⚖️Critic评分器", args.training_config.critic_learning_rate, critic_accelerator)

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, accelerator, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)
    critic_optimizer = get_optimizer(args, critic_accelerator, critic_params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    dataset_kwargs = {
        "gan_folders": args.data_config.gan_data_root if args.training_config.is_use_gt_history else None,
        "ode_folders": None,
        "text_folders": args.data_config.text_data_root
        if not args.training_config.is_only_ode_regression
        else None,
        "is_use_gt_history": args.training_config.is_use_gt_history,
        "return_secondary": args.training_config.is_use_gt_history,
        "single_res": args.data_config.single_res,
        "single_length": args.data_config.single_length,
        "single_num_frame": args.data_config.single_num_frame,
        "single_height": args.data_config.single_height,
        "single_width": args.data_config.single_width,
        "force_rebuild": args.data_config.force_rebuild,
        "seed": args.seed,
    }
    assert any(
        [
            dataset_kwargs["gan_folders"],
            dataset_kwargs["ode_folders"],
            dataset_kwargs["text_folders"],
        ]
    ), "Invalid dataset config: at least one of gan_folders/ode_folders/text_folders must be non-empty."
    if args.training_config.is_use_gt_history:
        assert dataset_kwargs["gan_folders"] is not None, "GT-history DMD needs data_config.gan_data_root."

    train_dataset = BucketedFeatureDataset(**dataset_kwargs)
    sampler = BucketedSampler(
        train_dataset,
        batch_size=args.training_config.train_batch_size,
        drop_last=True,
        shuffle=args.data_config.use_shuffle,
        seed=args.seed,
        dataset_sampling_ratios={},
        num_sp_groups=accelerator.num_processes,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_sampler=sampler,
        pin_memory=args.data_config.pin_memory,
        prefetch_factor=args.data_config.prefetch_factor if args.data_config.prefetch_factor > 0 else None,
        persistent_workers=args.data_config.persistent_workers,
        collate_fn=collate_fn,
        num_workers=args.data_config.dataloader_num_workers,
    )

    print_dataset_info(accelerator, dataset_kwargs, train_dataset, train_dataloader, sampler)

    if args.model_config.load_dcp:
        dcp_dir = os.path.join(
            args.model_config.load_dcp_path or args.model_config.load_model_path,
            "distributed_checkpoint",
        )
        dcp.load({"dataloader": train_dataloader}, checkpoint_id=dcp_dir)
        rank0_print(accelerator, f"Loaded dcp from {dcp_dir}")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training_config.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = args.training_config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.training_config.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
        )
        critic_lr_scheduler = DummyScheduler(
            name=args.training_config.lr_scheduler,
            optimizer=critic_optimizer,
            total_num_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.training_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_cycles=args.training_config.lr_num_cycles,
            power=args.training_config.lr_power,
        )
        critic_lr_scheduler = get_scheduler(
            args.training_config.lr_scheduler,
            optimizer=critic_optimizer,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_cycles=args.training_config.lr_num_cycles,
            power=args.training_config.lr_power,
        )

    accelerator.wait_for_everyone()
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.select_deepspeed_plugin("generator")
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            args.training_config.train_batch_size
        )
    
    # ============================================================
    # EMA CPU copy must be created before accelerator.prepare()
    # ============================================================
    global_step = 0
    first_epoch = 0
    ema_transformer = None
    transformer_cpu = None
    ds_config = None
    model_cls = GigaworldTransformer3DModelFunCtrl
    vram_manager = OptimizedLowVRAMManager() if args.training_config.dmd_is_low_vram_mode else None

    if args.training_config.use_ema:
        transformer_cpu = copy.deepcopy(transformer).to("cpu")
        with open(args.training_config.ema_deepspeed_config_file, "r") as f:
            ds_config = json.load(f)

    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    if dmd_deepspeed_training:
        critic_accelerator.state.select_deepspeed_plugin("critic_model")
        critic_accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            args.training_config.train_batch_size
        )
    real_score_model, critic_optimizer, critic_lr_scheduler = critic_accelerator.prepare(
        real_score_model, critic_optimizer, critic_lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training_config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.training_config.max_train_steps = args.training_config.num_train_epochs * num_update_steps_per_epoch
    args.training_config.num_train_epochs = math.ceil(args.training_config.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        from omegaconf import OmegaConf

        tracker_name = args.report_to.tracker_name or "wanvideo-dmd-train"
        wandb_name = args.report_to.wandb_name or "dmd-only"
        accelerator.init_trackers(
            tracker_name,
            config=OmegaConf.to_container(args, resolve=True),
            init_kwargs={"wandb": {"name": wandb_name, "dir": args.output_dir}},
        )

    total_batch_size = (
        args.training_config.train_batch_size
        * accelerator.num_processes
        * args.training_config.gradient_accumulation_steps
    )
    num_trainable_parameters = sum(param.numel() for group in params_to_optimize for param in group["params"])
    critic_num_trainable_parameters = sum(param.numel() for group in critic_params_to_optimize for param in group["params"])

    print_model_info(accelerator, "Generator Transformer", accelerator.unwrap_model(transformer))
    print_model_info(accelerator, "Critic / Fake Score Model", critic_accelerator.unwrap_model(real_score_model))

    accelerator.print("🚀***** Running DMD-only training *****")
    accelerator.print(f"  🧩 Generator trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  ⚖️ Critic trainable parameters = {critic_num_trainable_parameters}")
    accelerator.print(f"  📚 Num examples = {len(train_dataset)}")
    accelerator.print(f"  📦 Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  🔁 Num Epochs = {args.training_config.num_train_epochs}")
    accelerator.print(f"  📏 Batch size per device = {args.training_config.train_batch_size}")
    accelerator.print(f"  📈 Total train batch size = {total_batch_size}")
    accelerator.print(f"  ⏳ Gradient accumulation steps = {args.training_config.gradient_accumulation_steps}")
    accelerator.print(f"  🎯 Total optimization steps = {args.training_config.max_train_steps}")


    if args.training_config.resume_from_checkpoint:
        if args.training_config.resume_from_checkpoint != "latest":
            resume_path = args.training_config.resume_from_checkpoint
            path = resume_path if os.path.isabs(resume_path) else os.path.join(args.output_dir, resume_path)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if dirs else None

        if path is None or not os.path.exists(path):
            accelerator.print(f"[ERROR ❌] Checkpoint '{args.training_config.resume_from_checkpoint}' does not exist. Starting new run.")
            args.training_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"[INFO ♻️] Resuming from checkpoint {path}")
            accelerator.load_state(path, load_kwargs={"weights_only": False})
            critic_accelerator.load_state(os.path.join(path, "critic"), load_kwargs={"weights_only": False})
            global_step = int(os.path.basename(path).split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            if args.training_config.use_ema:
                if args.training_config.dmd_is_low_vram_mode:
                    vram_manager.move_to_cpu(transformer, non_blocking=False)
                    vram_manager.move_to_cpu(real_score_model, non_blocking=False)

                transformer_cpu.load_state_dict(unwrap_model(transformer).state_dict())
                ema_transformer = create_ema_final(
                    accelerator=accelerator,
                    args=args,
                    transformer_cpu=transformer_cpu,
                    model_cls=model_cls,
                    ds_config=ds_config,
                    transformer_lora_config=transformer_lora_config,
                    resume_checkpoint_path=os.path.join(path, "model_ema"),
                    transformer_additional_kwargs=transformer_additional_kwargs,
                )
                accelerator.wait_for_everyone()
                transformer_cpu = None
                del transformer_cpu

                if args.training_config.dmd_is_low_vram_mode:
                    vram_manager.move_to_gpu(transformer, accelerator.device)
                    vram_manager.move_to_gpu(real_score_model, accelerator.device)
    else:
        initial_global_step = 0

    if args.model_config.load_checkpoints_custom:
        assert initial_global_step == 0

    progress_bar = tqdm(
        range(0, args.training_config.max_train_steps),
        initial=initial_global_step,
        desc="🚀 Training Steps",
        disable=not accelerator.is_local_main_process,
        colour='green',
        bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'
    )

    if ema_transformer is None and args.training_config.use_ema:
        if args.training_config.dmd_is_low_vram_mode:
            vram_manager.move_to_cpu(transformer, non_blocking=False)
            vram_manager.move_to_cpu(real_score_model, non_blocking=False)
        else:
            transformer.to("cpu", non_blocking=False)

        transformer_cpu.load_state_dict(unwrap_model(transformer).state_dict())
        ema_transformer = create_ema_final(
            accelerator=accelerator,
            args=args,
            transformer_cpu=transformer_cpu,
            model_cls=model_cls,
            ds_config=ds_config,
            transformer_lora_config=transformer_lora_config,
            update_after_step=args.training_config.ema_start_step,
        )
        accelerator.wait_for_everyone()
        transformer_cpu = None
        del transformer_cpu

        if args.training_config.dmd_is_low_vram_mode:
            vram_manager.move_to_gpu(transformer, accelerator.device)
            vram_manager.move_to_gpu(real_score_model, accelerator.device)
        else:
            transformer.to(accelerator.device, non_blocking=False)

    # Pure DMD: no GAN branch and no GT-history branch.
    gan_critic_trainable_params = None
    gan_extra_critic_trainable_params = None
    gan_base_critic_trainable_params = None

    accelerator.wait_for_everyone()

    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        transformer.train()
        real_score_model.train()
        train_dataset.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, real_score_model]

            with torch.no_grad():
                latent_window_size = args.training_config.latent_window_size[0]
                if args.model_config.model_type in ["wan2.1", "wan2.2"]:
                    noisy_model_input_shape = (
                        args.training_config.train_batch_size,
                        16,
                        latent_window_size,
                        args.data_config.single_height // 8,
                        args.data_config.single_width // 8,
                    )
                elif args.model_config.model_type == "wan2.2_5b":
                    noisy_model_input_shape = (
                        args.training_config.train_batch_size,
                        48,
                        latent_window_size,
                        args.data_config.single_height // 16,
                        args.data_config.single_width // 16,
                    )

                #prompt_raws = batch["text_prompt_raws"]
                # prompt_embeds = batch["text_prompt_embeds"].to(
                #     accelerator.device, dtype=weight_dtype, non_blocking=True
                # )

                gan_vae_latents = None
                gan_prompt_embeds = None
                gt_history_latents = None
                gt_target_latents = None
                gt_x0_latents = None
                gt_history_latents_2 = None
                gt_target_latents_2 = None
                gt_x0_latents_2 = None

                if args.training_config.is_use_gt_history:
                    gan_vae_latents = batch["gan_vae_latents"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    gan_prompt_embeds = batch["gan_prompt_embeds"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    prompt_raws = batch["gan_prompt_raws"]
                    prompt_embeds = gan_prompt_embeds

                    gt_target_latents = batch["gan_vae_latents"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    gt_x0_latents = batch["gan_x0_latents"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    gt_history_latents = batch["gan_history_latents"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )

                    gt_target_latents_2 = batch["gan_vae_latents_2"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    gt_x0_latents_2 = batch["gan_x0_latents_2"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    gt_history_latents_2 = batch["gan_history_latents_2"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    gt_control_latents = batch["gan_control_latents"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )

                    gt_control_latents_2 = batch["gan_control_latents_2"].to(
                        accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    assert gt_target_latents_2.shape[2] == args.training_config.num_critic_input_frames

                batch = None
                del batch

            with accelerator.accumulate(models_to_accumulate):
                TRAIN_GENERATOR = global_step % args.training_config.dfake_gen_update_ratio == 0
                USE_GAN = False
                USE_GT_HIST = (
                    args.training_config.is_use_gt_history
                    and random.random() < args.training_config.use_gt_history_ratio
                )
                VISUALIZE = global_step % args.training_config.log_iters == 0 and not args.training_config.no_visualize
                logs = {}

                if accelerator.is_main_process:
                    if args.training_config.is_enable_cold_start and global_step < args.training_config.cold_start_step:
                        num_rollout_sections = (
                            args.training_config.dmd_num_latent_sections_min + 1
                            if args.training_config.stage_cold_start_step is not None
                            and global_step >= args.training_config.stage_cold_start_step
                            else args.training_config.dmd_num_latent_sections_min
                        )
                    else:
                        num_rollout_sections = sample_dynamic_dmd_num_latent_sections(
                            min_sections=args.training_config.dmd_num_latent_sections_min,
                            max_sections=args.training_config.dmd_num_latent_sections_max,
                            dmd_dynamic_alpha=args.training_config.dmd_dynamic_alpha,
                            dmd_dynamic_beta=args.training_config.dmd_dynamic_beta,
                            dmd_dynamic_sample_type=args.training_config.dmd_dynamic_sample_type,
                            global_step=global_step,
                            dmd_dynamic_step=args.training_config.dmd_dynamic_step,
                            device=accelerator.device,
                        )
                    num_rollout_sections = torch.tensor(num_rollout_sections, device=accelerator.device)
                else:
                    num_rollout_sections = torch.tensor(0, device=accelerator.device)

                num_rollout_sections = broadcast(num_rollout_sections, from_process=0).item()
                logs["num_rollout_sections"] = num_rollout_sections
 
                if TRAIN_GENERATOR: 
                    generator_loss, generator_log_dict = _generator_loss(
                        args=args,
                        accelerator=accelerator,
                        real_fake_score_model=real_score_model,
                        transformer=transformer,
                        scheduler=noise_scheduler_copy,
                        noise=torch.randn(noisy_model_input_shape, device=accelerator.device, dtype=weight_dtype),
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        dmd_is_low_vram_mode=args.training_config.dmd_is_low_vram_mode,
                        vram_manager=vram_manager,
                        dmd_is_offload_grad=args.training_config.dmd_is_offload_grad,
                        is_gan_low_vram_mode=False,
                        is_keep_x0=True,
                        history_sizes=args.training_config.history_sizes,
                        denoising_step_list=list(args.training_config.dmd_denoising_step_list),
                        last_step_only=args.training_config.dmd_last_step_only,
                        last_section_grad_only=args.training_config.dmd_last_section_grad_only,
                        timestep_shift=args.training_config.dmd_timestep_shift,
                        use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                        time_shift_type=args.training_config.time_shift_type,
                        fake_guidance_scale=args.training_config.fake_guidance_scale,
                        real_guidance_scale=args.training_config.real_guidance_scale,
                        num_critic_input_frames=args.training_config.num_critic_input_frames,
                        num_rollout_sections=num_rollout_sections,
                        is_skip_first_section=args.training_config.is_skip_first_section,
                        is_amplify_first_chunk=args.training_config.is_amplify_first_chunk,
                        is_corrupt_history_latents=args.training_config.corrupt_history,
                        is_add_saturation=args.training_config.is_add_saturation,
                        is_use_gt_history=USE_GT_HIST,
                        gt_history_latents=gt_history_latents,
                        gt_target_latents=gt_target_latents,
                        gt_x0_latents=gt_x0_latents,
                        # FunCtrl control
                        gt_control_latents=gt_control_latents,
                        vae=vae,
                        is_dmd_vae_decode=args.training_config.is_dmd_vae_decode,
                        is_multi_pyramid_stage_backward_simulated=args.training_config.is_multi_pyramid_stage_backward_simulated,
                        is_consistency_align=args.training_config.is_consistency_align,
                        consistentcy_align_weight=args.training_config.consistentcy_align_weight,
                        is_smoothness_loss=args.training_config.is_smoothness_loss,
                        smoothness_loss_weight=args.training_config.smoothness_loss_weight,
                        use_kv_cache=args.validation_config.use_kv_cache,
                        is_mean_var_regular=args.training_config.is_mean_var_regular,
                        mean_var_regular_weight=args.training_config.mean_var_regular_weight,
                        regular_mean=args.training_config.regular_mean,
                        regular_var=args.training_config.regular_var,
                        is_x0_mean_var_regular=args.training_config.is_x0_mean_var_regular,
                        mean_var_regular_x0_weight=args.training_config.mean_var_regular_x0_weight,
                        regular_x0_mean=args.training_config.regular_x0_mean,
                        regular_x0_var=args.training_config.regular_x0_var,
                        is_chunk_mean_var_regular=args.training_config.is_chunk_mean_var_regular,
                        chunk_mean_var_regular_weight=args.training_config.chunk_mean_var_regular_weight,
                        chunk_regular_mean=args.training_config.chunk_regular_mean,
                        chunk_regular_var=args.training_config.chunk_regular_var,
                        is_chunk_x0_mean_var_regular=args.training_config.is_chunk_x0_mean_var_regular,
                        chunk_mean_var_regular_x0_weight=args.training_config.chunk_mean_var_regular_x0_weight,
                        chunk_regular_x0_mean=args.training_config.chunk_regular_x0_mean,
                        chunk_regular_x0_var=args.training_config.chunk_regular_x0_var,
                        is_use_gan=False,
                        gan_prompt_embeds=prompt_embeds, # FIXME: ORI: None. DEBUG FOR GIGA, REMOVE LATER
                        gan_g_weight=0.0,
                        is_use_reward_model=False,
                        reward_model=None,
                        reward_weight_vq=0,
                        reward_weight_mq=0,
                        reward_weight_ta=0,
                        reward_texts=prompt_raws,
                        is_decouple_dmd=args.training_config.is_decouple_dmd,
                        decouple_ca_start_step=args.training_config.decouple_ca_start_step,
                        decouple_ca_end_step=args.training_config.decouple_ca_end_step,
                        is_forcing_low_renoise=args.training_config.generator_is_forcing_low_renoise,
                        dynamic_alpha=args.training_config.generator_dynamic_alpha,
                        dynamic_beta=args.training_config.generator_dynamic_beta,
                        dynamic_sample_type=args.training_config.generator_dynamic_sample_type,
                        global_step=global_step,
                        dynamic_step=args.training_config.generator_dynamic_step,
                        model_type=args.model_config.model_type,
                    )

                    accelerator.backward(generator_loss)

                    generator_grad_norm = None
                    if accelerator.sync_gradients:
                        generator_grad_norm = accelerator.clip_grad_norm_(
                            transformer.parameters(), args.training_config.max_grad_norm
                        )

                    generator_log_dict["generator_loss"] = generator_loss
                    if generator_grad_norm is not None:
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm

                    generator_log_dict = merge_dict_list([generator_log_dict])
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    logs.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": safe_item(generator_log_dict["generator_grad_norm"]),
                        }
                    )
                    if args.training_config.is_decouple_dmd:
                        logs.update(
                            {
                                "dmdtrain_ca_gradient_norm": safe_item(generator_log_dict["dmdtrain_ca_gradient_norm"]),
                                "dmdtrain_dm_gradient_norm": safe_item(generator_log_dict["dmdtrain_dm_gradient_norm"]),
                            }
                        )
                    else:
                        logs["dmdtrain_gradient_norm"] = safe_item(generator_log_dict["dmdtrain_gradient_norm"])

                    if args.training_config.is_smoothness_loss:
                        logs["dmd_loss_raw"] = generator_log_dict["dmd_loss_raw"]
                    if args.training_config.is_consistency_align:
                        logs["consistency_align_loss"] = generator_log_dict["consistency_align_loss"]
                    if args.training_config.is_smoothness_loss:
                        logs["smoothness_loss"] = generator_log_dict["smoothness_loss"]
                    if args.training_config.is_mean_var_regular:
                        logs["kl_mean_var_loss"] = generator_log_dict["kl_mean_var_loss"]
                        logs["pred_mean_avg"] = generator_log_dict["pred_mean_avg"]
                        logs["pred_var_avg"] = generator_log_dict["pred_var_avg"]
                    if args.training_config.is_x0_mean_var_regular:
                        logs["kl_mean_var_x0_loss"] = generator_log_dict["kl_mean_var_x0_loss"]
                        logs["pred_x0_mean_avg"] = generator_log_dict["pred_x0_mean_avg"]
                        logs["pred_x0_var_avg"] = generator_log_dict["pred_x0_var_avg"]
                    if args.training_config.is_chunk_mean_var_regular:
                        logs["kl_chunk_mean_var_loss"] = generator_log_dict["kl_chunk_mean_var_loss"]
                        logs["pred_chunk_mean_avg"] = generator_log_dict["pred_chunk_mean_avg"]
                        logs["pred_chunk_var_avg"] = generator_log_dict["pred_chunk_var_avg"]
                    if args.training_config.is_chunk_x0_mean_var_regular:
                        logs["kl_chunk_mean_var_x0_loss"] = generator_log_dict["kl_chunk_mean_var_x0_loss"]
                        logs["pred_chunk_x0_mean_avg"] = generator_log_dict["pred_chunk_x0_mean_avg"]
                        logs["pred_chunk_x0_var_avg"] = generator_log_dict["pred_chunk_x0_var_avg"]
                    del generator_loss, generator_grad_norm
                    free_memory()

                critic_loss, critic_log_dict = _critic_loss(
                    args=args,
                    critic_accelerator=critic_accelerator,
                    fake_score_model=real_score_model,
                    transformer=transformer,
                    scheduler=critic_noise_scheduler,
                    noise=torch.randn(noisy_model_input_shape, device=critic_accelerator.device, dtype=weight_dtype),
                    prompt_embeds=prompt_embeds,
                    gan_prompt_embeds=prompt_embeds, # FIXME: ORI: None. DEBUG FOR GIGA, REMOVE LATER
                    dmd_is_low_vram_mode=args.training_config.dmd_is_low_vram_mode,
                    vram_manager=vram_manager,
                    is_gan_low_vram_mode=False,
                    is_keep_x0=True,
                    history_sizes=args.training_config.history_sizes,
                    denoising_step_list=list(args.training_config.dmd_denoising_step_list),
                    last_step_only=args.training_config.dmd_last_step_only,
                    last_section_grad_only=args.training_config.dmd_last_section_grad_only,
                    timestep_shift=args.training_config.dmd_timestep_shift,
                    use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                    time_shift_type=args.training_config.time_shift_type,
                    num_critic_input_frames=args.training_config.num_critic_input_frames,
                    num_rollout_sections=num_rollout_sections,
                    is_skip_first_section=args.training_config.is_skip_first_section,
                    is_amplify_first_chunk=args.training_config.is_amplify_first_chunk,
                    is_corrupt_history_latents=args.training_config.corrupt_history,
                    is_add_saturation=args.training_config.is_add_saturation,
                    is_use_gt_history=USE_GT_HIST,
                    gt_history_latents=gt_history_latents_2,
                    gt_target_latents=gt_target_latents_2,
                    gt_x0_latents=gt_x0_latents_2,
                    gt_control_latents=gt_control_latents_2, # NOTE: FunCtrl control for critic, only used when is_use_gt_history is True
                    vae=vae,
                    is_dmd_vae_decode=args.training_config.is_dmd_vae_decode,
                    is_multi_pyramid_stage_backward_simulated=args.training_config.is_multi_pyramid_stage_backward_simulated,
                    use_kv_cache=args.validation_config.use_kv_cache,
                    is_use_gan=False,
                    is_separate_gan_grad=args.training_config.is_separate_gan_grad,
                    gan_base_critic_trainable_params=None,
                    gan_extra_critic_trainable_params=None,
                    gan_vae_latents=None,
                    # gan_prompt_embeds=None,
                    gan_d_weight=0.0,
                    aprox_r1=args.training_config.aprox_r1,
                    aprox_r2=args.training_config.aprox_r2,
                    r1_weight=args.training_config.r1_weight,
                    r2_weight=args.training_config.r2_weight,
                    r1_sigma=args.training_config.r1_sigma,
                    r2_sigma=args.training_config.r2_sigma,
                    dynamic_alpha=args.training_config.critic_dynamic_alpha,
                    dynamic_beta=args.training_config.critic_dynamic_beta,
                    dynamic_sample_type=args.training_config.critic_dynamic_sample_type,
                    global_step=global_step,
                    dynamic_step=args.training_config.critic_dynamic_step,
                    model_type=args.model_config.model_type,
                )

                critic_accelerator.backward(critic_loss)

                critic_grad_norm = None
                if critic_accelerator.sync_gradients:
                    critic_grad_norm = critic_accelerator.clip_grad_norm_(
                        real_score_model.parameters(), args.training_config.max_grad_norm_critic
                    )

                critic_log_dict["critic_loss"] = critic_loss
                if critic_grad_norm is not None:
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm

                critic_log_dict = merge_dict_list([critic_log_dict])
                critic_optimizer.step()
                critic_lr_scheduler.step()
                critic_optimizer.zero_grad(set_to_none=True)

                logs.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": safe_item(critic_log_dict["critic_grad_norm"]),
                    }
                )
                del critic_loss, critic_grad_norm
                free_memory()

                prompt_embeds = None
                gan_vae_latents = None
                gan_prompt_embeds = None
                gt_history_latents = None
                gt_target_latents = None
                gt_x0_latents = None
                gt_history_latents_2 = None
                gt_target_latents_2 = None
                gt_x0_latents_2 = None
                free_memory()

            if accelerator.sync_gradients:
                if args.training_config.use_ema and ema_transformer is not None:
                    if global_step < args.training_config.ema_start_step or TRAIN_GENERATOR:
                        ema_transformer.step(transformer.parameters())

                progress_bar.update(1)
                global_step += 1

                optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)

                if "generator_log_dict" in locals():
                    generator_log_dict.clear()
                    del generator_log_dict
                if "critic_log_dict" in locals():
                    critic_log_dict.clear()
                    del critic_log_dict
                free_memory()

                if global_step % args.training_config.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    dcp.save({"dataloader": train_dataloader}, checkpoint_id=os.path.join(save_path, "distributed_checkpoint"))
                    free_memory()

                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        if args.training_config.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.training_config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.training_config.checkpoints_total_limit + 1
                                for removing_checkpoint in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                        accelerator.save_state(save_path)
                        if args.training_config.save_checkpoints_custom:
                            if accelerator.is_main_process:
                                save_model_checkpoint(
                                    transformer=transformer,
                                    args=args,
                                    save_path=save_path,
                                    weight_dtype=weight_dtype,
                                    unwrap_model_fn=unwrap_model,
                                    get_peft_model_state_dict_fn=get_peft_model_state_dict,
                                    collate_lora_metadata_fn=_collate_lora_metadata,
                                    save_extra_components_fn=save_extra_components,
                                    pipeline_class=GigaworldFunCtrlPipeline,
                                    norm_layer_prefixes=NORM_LAYER_PREFIXES,
                                )
                                save_model_checkpoint(
                                    transformer=real_score_model,
                                    args=args,
                                    save_path=os.path.join(save_path, "critic"),
                                    weight_dtype=weight_dtype,
                                    unwrap_model_fn=unwrap_model,
                                    get_peft_model_state_dict_fn=get_peft_model_state_dict,
                                    collate_lora_metadata_fn=_collate_lora_metadata,
                                    save_extra_components_fn=save_extra_components,
                                    pipeline_class=GigaworldFunCtrlPipeline,
                                    norm_layer_prefixes=NORM_LAYER_PREFIXES,
                                )
                        else:
                            accelerator.save_state(save_path)
                            critic_accelerator.save_state(os.path.join(save_path, "critic"))

                        accelerator.print(f"Saved state to {save_path}")

                    if args.training_config.use_ema and ema_transformer is not None:
                        ema_transformer.save_pretrained(
                            args,
                            os.path.join(save_path, "model_ema"),
                            args.model_config.transformer_model_name_or_path,
                            lora_config=transformer_lora_config,
                            transformer_additional_kwargs=transformer_additional_kwargs,
                        )

                if (
                    args.validation_config.validation_prompts is not None
                    and global_step % args.validation_config.validation_steps == 0
                ) or (args.validation_config.first_step_valid and global_step == (initial_global_step + 1)):
                    if args.training_config.dmd_is_low_vram_mode:
                        vram_manager.move_to_cpu(real_score_model)

                    optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)
                    free_memory()

                    if (
                        args.training_config.use_ema_validation
                        and args.training_config.use_ema
                        and ema_transformer is not None
                        and global_step >= args.training_config.ema_start_step
                    ):
                        accelerator.print("Starting EMA store and copy_to...")
                        ema_transformer.store(transformer.parameters())
                        ema_state_dict = gather_zero3ema(accelerator, ema_transformer)
                        transformer.load_state_dict({"module." + k: v for k, v in ema_state_dict.items()})
                        accelerator.print("EMA store and copy_to completed")
                        del ema_state_dict

                    if accelerator.is_main_process:
                        with torch.no_grad():

                            if vae is None:
                                vae = AutoencoderKLWan.from_pretrained(
                                    args.model_config.pretrained_model_name_or_path,
                                    subfolder="vae",
                                    revision=args.model_config.revision,
                                    variant=args.model_config.variant,
                                    torch_dtype=torch.float32,
                                    device_map=accelerator.device,
                                )

                                if args.model_config.enable_slicing:
                                    vae.enable_slicing()

                                if args.model_config.enable_tiling:
                                    vae.enable_tiling()

                            val_text_encoder = UMT5EncoderModel.from_pretrained(
                                args.model_config.pretrained_model_name_or_path,
                                subfolder="text_encoder",
                                revision=args.model_config.revision,
                                variant=args.model_config.variant,
                                dtype=weight_dtype,
                                device_map=accelerator.device,
                            )

                            run_validation_functrl(
                                args=args,
                                accelerator=accelerator,
                                transformer=unwrap_model(transformer),
                                tokenizer=tokenizer,
                                vae=vae,
                                text_encoder=val_text_encoder,
                                noise_scheduler=noise_scheduler,
                                weight_dtype=weight_dtype,
                                global_step=global_step,
                            )

                            del val_text_encoder
                            free_memory()

                    if (
                        args.training_config.use_ema_validation
                        and args.training_config.use_ema
                        and ema_transformer is not None
                        and global_step >= args.training_config.ema_start_step
                    ):
                        accelerator.wait_for_everyone()
                        ema_transformer.restore(transformer.parameters())

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.training_config.max_train_steps:
                break

            del logs
            free_memory()

        if global_step >= args.training_config.max_train_steps:
            break

    real_score_model.to("cpu", non_blocking=True)
    accelerator.wait_for_everyone()

    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-final")
    if args.training_config.use_ema and ema_transformer is not None:
        ema_transformer.save_pretrained(
            args,
            os.path.join(save_path, "model_ema"),
            args.model_config.transformer_model_name_or_path,
            lora_config=transformer_lora_config,
            transformer_additional_kwargs=transformer_additional_kwargs,
        )

    if accelerator.is_main_process:
        modules_to_save = {}
        model_to_save = unwrap_model(transformer)
        original_dtype = next(model_to_save.parameters()).dtype

        if args.model_config.bnb_quantization_config_path is None:
            if args.training_config.upcast_before_saving:
                model_to_save.to(torch.float32)
            else:
                model_to_save.to(weight_dtype)

        transformer_lora_layers = get_peft_model_state_dict(model_to_save)
        if args.model_config.train_norm_layers:
            transformer_norm_layers = {
                f"transformer.{name}": param
                for name, param in model_to_save.named_parameters()
                if any(k in name for k in NORM_LAYER_PREFIXES)
            }
            transformer_lora_layers = {**transformer_lora_layers, **transformer_norm_layers}

        modules_to_save["transformer"] = model_to_save
        GigaworldFunCtrlPipeline.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )
        save_extra_components(args, model=model_to_save, output_dir=save_path)
        model_to_save.to(original_dtype)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(Args)
    conf = OmegaConf.merge(schema, config)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != conf.training_config.local_rank:
        conf.training_config.local_rank = env_local_rank

    assert conf.training_config.is_train_dmd, "DMD-only script requires is_train_dmd=True."
    assert conf.data_config.use_stage2_dataset, "DMD-only script requires use_stage2_dataset=True."
    assert not conf.training_config.is_use_ode_regression, "DMD-only script removes ODE regression."

    if conf.model_config.lora_layers is not None:
        assert len(conf.model_config.lora_target_modules) == 0, (
            "lora_target_modules must be empty when lora_layers is set."
        )

    if conf.training_config.efficient_sample:
        assert conf.training_config.pyramid_sample_mode == "full", (
            f"efficient_sample requires pyramid_sample_mode='full', got {conf.training_config.pyramid_sample_mode}"
        )

    if conf.data_config.single_res:
        assert conf.data_config.force_rebuild, "force_rebuild must be True when single_res is enabled"

    if (
        conf.training_config.is_train_full_multi_term_memory_patchg
        or conf.training_config.is_train_lora_multi_term_memory_patchg
        or conf.training_config.zero_history_timestep
    ):
        assert conf.training_config.has_multi_term_memory_patch, "Missing multi-term memory patch config."
        assert conf.training_config.is_enable_stage1, "is_enable_stage1 must be enabled."

    if conf.training_config.restrict_lora:
        assert conf.training_config.restrict_self_attn, "restrict_lora requires restrict_self_attn."

    if conf.training_config.is_train_restrict_lora:
        assert conf.training_config.restrict_lora, "is_train_restrict_lora requires restrict_lora."

    assert not (
        conf.training_config.is_train_full_multi_term_memory_patchg
        and conf.training_config.is_train_lora_multi_term_memory_patchg
    ), "Cannot train full and LoRA multi-term memory patches at the same time."

    assert not (
        conf.training_config.is_train_full_patch_embedding
        and conf.training_config.is_train_lora_patch_embedding
    ), "Cannot train full and LoRA patch embedding at the same time."

    assert not (
        conf.training_config.use_error_recycling and conf.training_config.corrupt_history
    ), "use_error_recycling and corrupt_history cannot both be True."

    assert not (
        conf.training_config.use_error_recycling and conf.training_config.corrupt_model_input
    ), "use_error_recycling and corrupt_model_input cannot both be True."

    if conf.validation_config.use_kv_cache:
        assert conf.training_config.restrict_self_attn, "use_kv_cache=True requires restrict_self_attn=True."

    if conf.training_config.use_ema_validation:
        assert conf.training_config.use_ema, "EMA validation requires use_ema."

    assert not conf.training_config.is_use_gan, "Pure DMD script requires is_use_gan=False."
    assert not conf.training_config.is_use_reward_model, "Pure DMD script requires is_use_reward_model=False."

    if conf.training_config.stage_cold_start_step is not None:
        assert conf.training_config.stage_cold_start_step <= conf.training_config.cold_start_step, (
            "stage_cold_start_step must be <= cold_start_step."
        )

    if conf.training_config.is_decouple_dmd:
        assert conf.training_config.decouple_ca_start_step >= conf.training_config.generator_dynamic_step
        assert conf.training_config.decouple_ca_end_step >= conf.training_config.generator_dynamic_step

    main(conf)
