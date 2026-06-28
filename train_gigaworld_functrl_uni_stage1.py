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

# ANSI colors
_C = {
    "dim": "\033[90m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
}

import argparse
import hashlib
import json
import logging
import math
from datetime import timedelta

# Silence accelerate checkpointing INFO logs (Saving state, scheduler, random_states, etc.)
logging.getLogger("accelerate.checkpointing").setLevel(logging.WARNING)
logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)
from pathlib import Path
import cv2
from PIL import Image

import torch
import torch.distributed.checkpoint as dcp
import transformers
import diffusers
import numpy as np
from peft import PeftModel
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

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
    compare_configs,
    encode_prompt,
    get_optimizer,
    load_extra_components,
    load_model_checkpoint,
    save_extra_components,
    save_model_checkpoint,
)
from gigaworld.utils.utils_gigaworld_base import (
    _flow_loss,
    prepare_stage1_clean_input_from_latents,
    prepare_stage1_noise_input,
)
from gigaworld.utils.utils_recycle_batch import get_timesteps
from diffusers.utils import export_to_video, load_image, load_video
if is_wandb_available():
    import wandb

check_min_version("0.36.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

def safe_item(x):
    return x.item() if hasattr(x, "item") else x


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
    if not accelerator.is_main_process:
        return

    if args.validation_config.validation_prompts is None:
        return

    vae.to(accelerator.device, non_blocking=True)
    text_encoder.to(accelerator.device, non_blocking=True)

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

    control_video_path = args.validation_config.validation_control_video[0]

    run_infer_dir = os.path.join(args.output_dir, "run_infer")
    os.makedirs(run_infer_dir, exist_ok=True)

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

    saved_files = []
    saved_prompts = []

    for sample_idx in range(args.validation_config.num_validation_videos):
        gen_video_ori = pipe(
            **pipeline_args,
            generator=generator,
            output_type="np",
        ).frames[0]

        prompt = pipeline_args["prompt"]
        safe_prompt = prompt[:25].replace(" ", "_").replace("/", "_")
        gen_filename = os.path.join(
            run_infer_dir,
            f"global_step{global_step}_control_gt_gen_{sample_idx}_{safe_prompt}.mp4",
        )

        export_to_video(gen_video_ori, gen_filename, fps=10)
        saved_files.append(gen_filename)
        saved_prompts.append(prompt)
        accelerator.print(f"✅ Saved validation video: {gen_filename}")

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            video_logs = [
                wandb.Video(
                    filename,
                    caption=(
                        f"{i}: generated "
                        f"| prompt={saved_prompts[i]} "
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

def make_functrl_input(
    noise_or_latents,
    control_latents,
    model_type: str,
    image_latents=None,
    mask_latents=None,
):
    if isinstance(noise_or_latents, list):
        return [
            make_functrl_input(
                x,
                control_latents,
                model_type=model_type,
                image_latents=image_latents,
                mask_latents=mask_latents,
            )
            for x in noise_or_latents
        ]

    # =====================================================
    # Wan2.1
    # =====================================================
    if model_type in ("wan2.1", "wan2.2"):

        if image_latents is None:
            image_latents = torch.zeros_like(noise_or_latents)

        if control_latents.shape[-3:] != noise_or_latents.shape[-3:]:
            control_latents = control_latents[:, :, : noise_or_latents.shape[2]]

        return torch.cat(
            [
                noise_or_latents,   # 16
                control_latents,    # 16
                image_latents,      # 16
            ],
            dim=1,
        )

    # =====================================================
    # Wan2.2 / Wan2.2-5B
    # =====================================================
    elif model_type == "wan2.2_5b":

        if image_latents is None:
            image_latents = torch.zeros_like(noise_or_latents)

        if mask_latents is None:
            mask_latents = torch.zeros(
                noise_or_latents.shape[0],
                4,
                noise_or_latents.shape[2],
                noise_or_latents.shape[3],
                noise_or_latents.shape[4],
                device=noise_or_latents.device,
                dtype=noise_or_latents.dtype,
            )

        if control_latents.shape[-3:] != noise_or_latents.shape[-3:]:
            control_latents = control_latents[:, :, : noise_or_latents.shape[2]]

        return torch.cat(
            [
                noise_or_latents,   # 48
                control_latents,    # 48
                mask_latents,       # 4
                image_latents,      # 48
            ],
            dim=1,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def main(args):
    # ============================================================
    # ✅ Stage1-only checks
    # ============================================================
    assert args.data_config.use_stage1_dataset, "❌ This script only supports use_stage1_dataset=True"
    assert not args.data_config.use_stage2_dataset, "❌ Stage2 removed"
    assert not args.training_config.is_train_dmd, "❌ DMD removed"
    assert not args.training_config.is_use_ode_regression, "❌ ODE removed"
    assert not args.training_config.is_use_gan, "❌ GAN removed"

    from gigaworld.dataset.dataloader_history_latents_dist import (
        BucketedFeatureDataset,
        BucketedSampler,
        collate_fn,
    )

    # ============================================================
    # 🚀 Accelerator
    # ============================================================
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(
        backend="nccl",
        timeout=timedelta(seconds=1800),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.training_config.gradient_accumulation_steps,
        mixed_precision=args.training_config.mixed_precision,
        log_with=args.report_to.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        config_path = os.path.join(args.output_dir, "config.json")
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

    if args.report_to.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("wandb is not installed.")

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

    # ============================================================
    # 🎚️ dtype
    # ============================================================
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    accelerator.print(f"🚀 weight_dtype = {weight_dtype}")

    # ============================================================
    # 🧠 tokenizer / scheduler / vae / text_encoder
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.model_config.revision,
    )

    noise_scheduler = UniPCMultistepScheduler.from_pretrained(
        "scripts/accelerate_configs/scheduler_config.json"
    )
    noise_scheduler_copy = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000
    )

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

    # ============================================================
    # 🏗️ transformer
    # ============================================================
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

    # ============================================================
    # ❄️ freeze base
    # ============================================================
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae.eval()
    text_encoder.eval()

    # ============================================================
    # 🧬 LoRA
    # ============================================================
    if args.model_config.lora_layers is not None:
        if args.model_config.lora_layers != "all-linear":
            target_modules = [
                layer.strip()
                for layer in args.model_config.lora_layers.split(",")
            ]

            if (
                args.training_config.is_train_lora_patch_embedding
                and "patch_embedding" not in target_modules
            ):
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

            if (
                args.training_config.is_train_lora_patch_embedding
                and "patch_embedding" not in target_modules
            ):
                target_modules.append("patch_embedding")

            if args.training_config.is_train_lora_multi_term_memory_patchg:
                for patch_name in ["patch_short", "patch_mid", "patch_long"]:
                    if patch_name not in target_modules:
                        target_modules.append(patch_name)

        # 不给 norm 加 LoRA
        target_modules = [
            t for t in target_modules
            if "norm" not in t
        ]

    else:
        target_modules = list(args.model_config.lora_target_modules)

    # ============================================================
    # LoRA exclude modules
    # ============================================================
    lora_exclude_modules = list(args.model_config.lora_exclude_modules)

    transformer_lora_config = LoraConfig(
        r=args.model_config.lora_rank,
        lora_alpha=args.model_config.lora_alpha,
        lora_dropout=args.model_config.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
        exclude_modules=lora_exclude_modules,
    )

    transformer.add_adapter(transformer_lora_config)

    # ============================================================
    # 🧩 trainable params
    # ============================================================
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # ============================================================
    # 🧩 print trainable params
    # ============================================================
    trainable_param_names = []
    trainable_params = 0
    all_params = 0
    for name, param in transformer.named_parameters():

        numel = param.numel()
        all_params += numel

        if param.requires_grad:
            trainable_param_names.append(name)
            trainable_params += numel

    accelerator.print(
        f"{_C['cyan']}[Trainable]{_C['reset']} "
        f"{trainable_params / 1e6:.2f}M / {all_params / 1e6:.2f}M "
        f"({100.0 * trainable_params / all_params:.4f}%) "
        f"across {len(trainable_param_names)} params"
    )
      
    # ============================================================
    # 🔁 load checkpoint
    # ============================================================
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

    # ============================================================
    # 🚚 device
    # ============================================================
    vae.to("cpu")
    text_encoder.to("cpu")
    free_memory()

    for name, param in transformer.named_parameters():
        should_keep_fp32 = any(
            pattern in name
            for pattern in transformer.__class__._keep_in_fp32_modules
        )
        if should_keep_fp32:
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(weight_dtype)

    transformer.to(accelerator.device)
    free_memory()

    if args.training_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.training_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ============================================================
    # 💾 hooks
    # ============================================================
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
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
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                if weights:
                    weights.pop()

            GigaworldFunCtrlPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

            save_extra_components(
                args,
                model=unwrap_model(model),
                output_dir=output_dir,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_ = model
                else:
                    raise ValueError(f"unexpected load model: {model.__class__}")
        else:
            transformer_ = GigaworldTransformer3DModelFunCtrl.from_pretrained(
                args.model_config.transformer_model_name_or_path,
                subfolder=args.model_config.subfolder or "transformer",
                transformer_additional_kwargs=transformer_additional_kwargs,
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = GigaworldFunCtrlPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }

        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)

        # incompatible_keys = set_peft_model_state_dict(
        #     transformer_,
        #     transformer_state_dict,
        #     adapter_name="default",
        # )

        missing, unexpected = transformer_.load_state_dict(
            transformer_state_dict,
            strict=False,
        )

        logger.warning(f"LoRA load missing keys: {len(missing)}")
        logger.warning(f"LoRA load unexpected keys: {len(unexpected)}")

        # if incompatible_keys is not None:
        #     unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        #     if unexpected_keys:
        #         logger.warning(f"Unexpected keys when loading adapter: {unexpected_keys}")

        if args.model_config.train_norm_layers:
            transformer_norm_state_dict = {
                k: v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
                and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
            }
            transformer_._transformer_norm_layers = GigaworldFunCtrlPipeline._load_norm_into_transformer(
                transformer_norm_state_dict,
                transformer=transformer_,
                discard_original_layers=False,
            )

        load_extra_components(
            args,
            transformer_,
            os.path.join(input_dir, "transformer_partial.pth"),
        )

        if args.training_config.mixed_precision != "fp32":
            cast_training_params([transformer_])

        dcp_dir = os.path.join(input_dir, "distributed_checkpoint")
        if os.path.exists(dcp_dir):
            states = {"dataloader": train_dataloader}
            dcp.load(states, checkpoint_id=dcp_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ============================================================
    # ⚡ TF32
    # ============================================================
    if args.training_config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.training_config.scale_lr:
        args.training_config.learning_rate = (
            args.training_config.learning_rate
            * args.training_config.gradient_accumulation_steps
            * args.training_config.train_batch_size
            * accelerator.num_processes
        )

    if args.training_config.mixed_precision != "fp32":
        cast_training_params([transformer], dtype=torch.float32)

    # ============================================================
    # 🧮 optimizer
    # ============================================================
    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    params_to_optimize = [
        {
            "params": trainable_params,
            "lr": args.training_config.learning_rate,
        }
    ]

    num_trainable_parameters = sum(p.numel() for p in trainable_params)

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        args,
        accelerator,
        params_to_optimize,
        use_deepspeed=use_deepspeed_optimizer,
    )

    # ============================================================
    # 📦 dataset
    # ============================================================
    dataset_sampling_ratios = {}
    if args.data_config.dataset_sampling_ratios:
        for root, ratio in zip(
            args.data_config.instance_data_root,
            args.data_config.dataset_sampling_ratios,
        ):
            dataset_sampling_ratios[root.rstrip("/")] = ratio

    dataset_kwargs = {
        "filter_tasks": args.data_config.filter_tasks,
        "feature_folders": args.data_config.instance_data_root,
        "single_res": args.data_config.single_res,
        "single_height": args.data_config.single_height,
        "single_width": args.data_config.single_width,
        "return_prompt_raw": False,
        "return_all_vae_latent": False,
        "history_sizes": args.training_config.history_sizes,
        "is_keep_x0": True,
        "force_rebuild": args.data_config.force_rebuild,
        "seed": args.seed,
        "is_control_model": args.model_config.is_control_model,
    }

    train_dataset = BucketedFeatureDataset(**dataset_kwargs)

    sampler = BucketedSampler(
        train_dataset,
        batch_size=args.training_config.train_batch_size,
        drop_last=True,
        shuffle=args.data_config.use_shuffle,
        seed=args.seed,
        dataset_sampling_ratios=dataset_sampling_ratios,
        num_sp_groups=accelerator.num_processes // 1,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_sampler=sampler,
        pin_memory=args.data_config.pin_memory,
        prefetch_factor=args.data_config.prefetch_factor
        if args.data_config.prefetch_factor > 0
        else None,
        persistent_workers=args.data_config.persistent_workers,
        collate_fn=collate_fn,
        num_workers=args.data_config.dataloader_num_workers,
    )

    if args.model_config.load_dcp:
        if args.model_config.load_dcp_path is not None:
            dcp_dir = os.path.join(args.model_config.load_dcp_path, "distributed_checkpoint")
        else:
            dcp_dir = os.path.join(args.model_config.load_model_path, "distributed_checkpoint")

        states = {"dataloader": train_dataloader}
        dcp.load(states, checkpoint_id=dcp_dir)
        accelerator.print(f"✅ Loaded dataloader DCP from {dcp_dir}")

    # ============================================================
    # 📉 lr scheduler
    # ============================================================
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )

    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.training_config.lr_scheduler,
            optimizer=optimizer,
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

    # ============================================================
    # 🚀 prepare
    # ============================================================
    accelerator.wait_for_everyone()

    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            args.training_config.train_batch_size
        )

    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer,
        optimizer,
        lr_scheduler,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )

    if overrode_max_train_steps:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )

    args.training_config.num_train_epochs = math.ceil(
        args.training_config.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        tracker_name = args.report_to.tracker_name or "wanvideo-train-stage1"
        wandb_name = args.report_to.wandb_name or "stage1-sft"
        accelerator.init_trackers(
            tracker_name,
            config=OmegaConf.to_container(args, resolve=True),
            init_kwargs={"wandb": {"name": wandb_name, "dir": args.output_dir}},
        )

    # ============================================================
    # 📊 train info
    # ============================================================
    total_batch_size = (
        args.training_config.train_batch_size
        * accelerator.num_processes
        * args.training_config.gradient_accumulation_steps
    )

    accelerator.print(
        f"{_C['cyan']}[Training]{_C['reset']} "
        f"steps={args.training_config.max_train_steps} "
        f"epochs={args.training_config.num_train_epochs} "
        f"bs/device={args.training_config.train_batch_size} "
        f"grad_accum={args.training_config.gradient_accumulation_steps} "
        f"total_bs={total_batch_size} "
        f"examples={len(train_dataset)} "
        f"batches/epoch={len(train_dataloader)} "
        f"trainable_params={num_trainable_parameters}"
    )

    global_step = 0
    first_epoch = 0

    # ============================================================
    # 🔁 resume
    # ============================================================
    if args.training_config.resume_from_checkpoint:
        if args.training_config.resume_from_checkpoint != "latest":
            resume_path = args.training_config.resume_from_checkpoint
            if not os.path.isabs(resume_path):
                resume_path = os.path.join(args.output_dir, resume_path)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if resume_path is None or not os.path.exists(resume_path):
            accelerator.print(f"⚠️ Checkpoint not found: {resume_path}, starting new run.")
            args.training_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"🔁 Resuming from checkpoint {resume_path}")
            accelerator.load_state(resume_path, load_kwargs={"weights_only": False}) #NOTE: TEMP BLOCK FOR DIFFERENT GPU CONFIG
            #accelerator.load_state(resume_path, load_kwargs={"weights_only": True})
            global_step = int(os.path.basename(resume_path).split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.training_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # optional recycle
    recycle_vars = None
    if args.training_config.use_error_recycling:
        from types import SimpleNamespace

        num_grids = args.training_config.num_grids

        recycle_vars = SimpleNamespace()
        recycle_vars.recycle_inferece_timesteps, recycle_vars.recycle_sigmas = get_timesteps(
            num_inference_steps=num_grids,
            denoising_strength=1,
            shift=1.0,
        )

        resolutions = set()
        for t, h, w in sampler.buckets.keys():
            base_h = h // 8
            base_w = w // 8
            resolutions.add((base_h, base_w))

        recycle_vars.latent_error_buffer = {
            resolution: {i: [] for i in range(num_grids)}
            for resolution in resolutions
        }
        recycle_vars.y_error_buffer = {
            resolution: {i: [] for i in range(num_grids)}
            for resolution in resolutions
        }

    accelerator.wait_for_everyone()

    # ============================================================
    # 🔥 train loop
    # ============================================================
    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        transformer.train()
        sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                latent_window_size = args.training_config.latent_window_size[0]

                prompt_embeds = batch["prompt_embeds"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )

                history_latents = batch["history_latents"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                target_latents = batch["target_latents"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                x0_latents = batch["x0_latents"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )

                control_target_latents = None
                if args.model_config.is_control_model:
                    control_target_latents = batch["control_target_latents"].to(
                        accelerator.device,
                        dtype=weight_dtype,
                        non_blocking=True,
                    )

                (
                    model_input,
                    indices_hidden_states,
                    indices_latents_history_short,
                    indices_latents_history_mid,
                    indices_latents_history_long,
                    latents_history_short,
                    latents_history_mid,
                    latents_history_long,
                ) = prepare_stage1_clean_input_from_latents(
                    history_latents=history_latents,
                    target_latents=target_latents,
                    x0_latents=x0_latents,
                    latent_window_size=latent_window_size,
                    history_sizes=args.training_config.history_sizes,
                    is_random_drop=args.training_config.is_random_drop,
                    random_drop_i2v_ratio=args.training_config.random_drop_i2v_ratio,
                    random_drop_v2v_ratio=args.training_config.random_drop_v2v_ratio,
                    random_drop_t2v_ratio=args.training_config.random_drop_t2v_ratio,
                    is_keep_x0=True,
                    dtype=weight_dtype,
                    device=accelerator.device,
                )

                del history_latents
                del target_latents
                del x0_latents

                dropout_mask = (
                    torch.rand(prompt_embeds.shape[0], device=prompt_embeds.device)
                    < args.data_config.caption_dropout_p
                )
                prompt_embeds[dropout_mask] = 0

                model_input = model_input.to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                indices_hidden_states = indices_hidden_states.to(
                    accelerator.device,
                    non_blocking=True,
                )
                indices_latents_history_short = indices_latents_history_short.to(
                    accelerator.device,
                    non_blocking=True,
                )
                indices_latents_history_mid = indices_latents_history_mid.to(
                    accelerator.device,
                    non_blocking=True,
                )
                indices_latents_history_long = indices_latents_history_long.to(
                    accelerator.device,
                    non_blocking=True,
                )
                latents_history_short = latents_history_short.to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                latents_history_mid = latents_history_mid.to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                latents_history_long = latents_history_long.to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                
                (
                    noisy_model_input_list,
                    sigmas_list,
                    timesteps_list,
                    targets_list,
                    latents_history_short,
                    latents_history_mid,
                    latents_history_long,
                    use_clean_input,
                ) = prepare_stage1_noise_input(
                    args=args,
                    model_input=model_input,
                    noise_scheduler=noise_scheduler_copy,
                    recycle_vars=recycle_vars,
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    latent_window_size=latent_window_size,
                    is_keep_x0=True,
                )

                if args.model_config.is_control_model:
                    assert control_target_latents is not None

                    noisy_model_input_list = [
                        make_functrl_input(
                            noisy_model_input,
                            control_target_latents,
                            model_type=args.model_config.model_type,
                            image_latents=None,
                        )
                        for noisy_model_input in noisy_model_input_list
                    ]

                    for i, x in enumerate(noisy_model_input_list):
                        if args.model_config.model_type == "wan2.2_5b":
                            assert x.shape[1] == 148, f"FunCtrl input must be 148 channels, got {x.shape}"
                        elif args.model_config.model_type == "wan2.1":
                            assert x.shape[1] == 48, f"FunCtrl input must be 148 channels, got {x.shape}"

            with accelerator.accumulate(transformer):
                assert len(noisy_model_input_list) == len(sigmas_list) == len(timesteps_list) == len(targets_list)
                logs = _flow_loss(
                    args=args,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                    transformer=transformer,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_masks=None,
                    noisy_model_input_list=noisy_model_input_list,
                    sigmas_list=sigmas_list,
                    timesteps_list=timesteps_list,
                    targets_list=targets_list,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    recycle_vars=recycle_vars,
                    global_step=global_step,
                    noise_scheduler_copy=noise_scheduler_copy,
                    use_clean_input=use_clean_input,
                )

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        transformer.parameters(),
                        args.training_config.max_grad_norm,
                    )
                    logs["grad_norm"] = safe_item(grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            del batch
            del model_input
            del prompt_embeds
            del indices_hidden_states
            del indices_latents_history_short
            del indices_latents_history_mid
            del indices_latents_history_long
            del latents_history_short
            del latents_history_mid
            del latents_history_long
            del noisy_model_input_list
            del sigmas_list
            del timesteps_list
            del targets_list
            free_memory()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.training_config.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    states = {"dataloader": train_dataloader}
                    dcp_dir = os.path.join(save_path, "distributed_checkpoint")
                    dcp.save(states, checkpoint_id=dcp_dir)

                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
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
                        else:
                            accelerator.save_state(save_path)

                        accelerator.print(f"{_C['green']}[Checkpoint]{_C['reset']} step={global_step} saved to {save_path}")
                        # ============================================================
                        # 💾 Save FULL transformer for inference
                        # ============================================================
                        if accelerator.is_main_process:
                            full_transformer_dir = os.path.join(
                                save_path,
                                "transformer_full",
                            )
                            os.makedirs(full_transformer_dir, exist_ok=True)

                        accelerator.wait_for_everyone()
                if (
                    args.validation_config.validation_prompts is not None
                    and global_step % args.validation_config.validation_steps == 0
                ) or (
                    args.validation_config.first_step_valid
                    and global_step == initial_global_step + 1
                ):  
                    if args.model_config.is_control_model:
                        run_validation_functrl(
                            args=args,
                            accelerator=accelerator,
                            transformer=unwrap_model(transformer),
                            tokenizer=tokenizer,
                            vae=vae,
                            text_encoder=text_encoder,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            global_step=global_step,
                        )
                    else:
                        raise NotImplementedError("Only control model validation is implemented for stage 1.")

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.training_config.max_train_steps:
                break

            del logs
            free_memory()

    # ============================================================
    # 💾 final save
    # ============================================================
    accelerator.wait_for_everyone()

    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-final")

    if accelerator.is_main_process:
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
            transformer_lora_layers = {
                **transformer_lora_layers,
                **transformer_norm_layers,
            }

        modules_to_save = {"transformer": model_to_save}

        GigaworldFunCtrlPipeline.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        save_extra_components(
            args,
            model=model_to_save,
            output_dir=save_path,
        )

        model_to_save.to(original_dtype)

        accelerator.print(f"{_C['green']}[Final]{_C['reset']} checkpoint saved to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    config = OmegaConf.load(cli_args.config)
    schema = OmegaConf.structured(Args)
    conf = OmegaConf.merge(schema, config)

    global_rank = int(os.environ.get("RANK", -1))
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if env_local_rank != -1 and env_local_rank != conf.training_config.local_rank:
        conf.training_config.local_rank = env_local_rank

    assert conf.data_config.use_stage1_dataset, "use_stage1_dataset must be True"
    assert not conf.data_config.use_stage2_dataset, "use_stage2_dataset must be False"
    assert not conf.training_config.is_train_dmd, "is_train_dmd must be False"
    assert not conf.training_config.is_use_ode_regression, "is_use_ode_regression must be False"
    assert not conf.training_config.is_use_gan, "is_use_gan must be False"

    assert (
        len(conf.validation_config.validation_latent_window_size) == 1
        and len(conf.validation_config.validation_stream_chunk_size) == 1
    ), "Only one validation_latent_window_size / validation_stream_chunk_size is supported."

    assert not (
        conf.data_config.use_stage1_dataset and conf.training_config.offload
    ), "use_stage1_dataset and offload cannot both be True"

    if conf.model_config.lora_layers is not None:
        assert len(conf.model_config.lora_target_modules) == 0, (
            f"lora_target_modules length is {len(conf.model_config.lora_target_modules)}, "
            "expected 0 when lora_layers is not None."
        )

    if conf.data_config.dataset_sampling_ratios:
        assert conf.data_config.use_stage1_dataset, (
            "dataset_sampling_ratios only supports use_stage1_dataset=True"
        )
        assert len(conf.data_config.instance_data_root) == len(conf.data_config.dataset_sampling_ratios)

    if conf.data_config.single_res:
        assert conf.data_config.force_rebuild, "force_rebuild must be True when single_res=True"

    if (
        conf.training_config.is_train_full_multi_term_memory_patchg
        or conf.training_config.is_train_lora_multi_term_memory_patchg
        or conf.training_config.zero_history_timestep
    ):
        assert conf.training_config.has_multi_term_memory_patch, "Missing multi-term memory patch config"
        assert conf.training_config.is_enable_stage1, "is_enable_stage1 must be True"

    if conf.training_config.restrict_lora:
        assert conf.training_config.restrict_self_attn, (
            "restrict_self_attn must be True when restrict_lora=True"
        )

    if conf.training_config.is_train_restrict_lora:
        assert conf.training_config.restrict_lora, (
            "restrict_lora must be True when is_train_restrict_lora=True"
        )

    assert not (
        conf.training_config.is_train_full_multi_term_memory_patchg
        and conf.training_config.is_train_lora_multi_term_memory_patchg
    ), "Cannot train full and LoRA multi-term memory patch together."

    assert not (
        conf.training_config.is_train_full_patch_embedding
        and conf.training_config.is_train_lora_patch_embedding
    ), "Cannot train full and LoRA patch_embedding together."

    assert not (
        conf.training_config.use_error_recycling and conf.training_config.corrupt_history
    ), "use_error_recycling and corrupt_history cannot both be True."

    if conf.validation_config.use_kv_cache:
        assert conf.training_config.restrict_self_attn, (
            "use_kv_cache=True requires restrict_self_attn=True"
        )

    main(conf)