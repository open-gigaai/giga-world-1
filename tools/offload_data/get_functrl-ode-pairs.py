import os
import sys
import time
import argparse
from pathlib import Path
import random

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"
sys.path.append("/mnt/pfs/users/zhanqian.wu/code/Helios/")
# sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

try:
    import peft.utils.save_and_load as peft_save_load

    def _noop_maybe_shard_state_dict_for_tp(model, peft_model_state_dict, adapter_name):
        return peft_model_state_dict

    peft_save_load._maybe_shard_state_dict_for_tp = _noop_maybe_shard_state_dict_for_tp
    print("✅ Patched PEFT tensor-parallel sharding")
except Exception as e:
    print(f"⚠️ Failed to patch PEFT TP sharding: {e}")

from accelerate import Accelerator
from diffusers.models import AutoencoderKLWan

from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.modules.transformer_functrl_helios import HeliosTransformer3DModelFunCtrl
from helios.pipelines.pipeline_helios_functrl_ode import HeliosFunCtrlPipeline
from helios.scheduler.scheduling_helios import HeliosScheduler
from helios.utils.utils_base import load_extra_components
from diffusers import FlowMatchEulerDiscreteScheduler

def is_dist_env():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ


def setup_distributed_env():
    if is_dist_env():
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()

    torch.cuda.set_device(0)
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def parse_pt_name(path):
    stem = Path(path).stem
    prefix, num_frames, height, width = stem.rsplit("_", 3)
    return prefix, int(num_frames), int(height), int(width)


def prepare_pt_dataset_on_rank0(input_pt_folder, output_folder, rank):
    if rank == 0:
        input_dir = Path(input_pt_folder)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        pt_paths = sorted(input_dir.glob("*.pt"))
        random.seed(42)
        random.shuffle(pt_paths)
        random.shuffle(pt_paths)
        random.shuffle(pt_paths)

        jobs = []
        skipped = 0
        broken = 0

        print("\n🔍 Scanning dataset...")
        scan_pbar = tqdm(
            pt_paths,
            desc="📂 Scan",
            dynamic_ncols=True,
            leave=True,
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for pt_path in scan_pbar:
            output_name = pt_path.name
            output_path = output_dir / output_name

            if output_path.exists():
                skipped += 1
            else:
                try:
                    uttid, num_frames, height, width = parse_pt_name(pt_path)
                    jobs.append(
                        {
                            "pt_path": str(pt_path),
                            "uttid": uttid,
                            "output_name": output_name,
                            "output_path": str(output_path),
                            "num_frames": num_frames,
                            "height": height,
                            "width": width,
                        }
                    )
                except Exception:
                    broken += 1

            scan_pbar.set_postfix_str(
                f"✅ exists={skipped:,} 🚧 todo={len(jobs):,} ⚠️ broken={broken:,}"
            )

        payload = {
            "jobs": jobs,
            "total": len(pt_paths),
            "skipped": skipped,
            "todo": len(jobs),
            "broken": broken,
        }
    else:
        payload = None

    if dist.is_available() and dist.is_initialized():
        obj = [payload]
        dist.broadcast_object_list(obj, src=0)
        payload = obj[0]

    return payload


class PtLatentDataset(Dataset):
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx):
        return self.jobs[idx]


def print_model_info(transformer, pipe=None, rank=0):
    if rank != 0:
        return

    print("\n================ Model Info ================")
    print(f"Transformer class: {transformer.__class__}")
    print(f"Transformer dtype: {next(transformer.parameters()).dtype}")
    print(f"Transformer device: {next(transformer.parameters()).device}")

    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

    print(f"total params: {total_params / 1e9:.3f}B")
    print(f"trainable params: {trainable_params / 1e6:.3f}M")

    if pipe is not None:
        print(f"Pipeline class: {pipe.__class__}")
        print(f"Scheduler class: {pipe.scheduler.__class__}")

    print("============================================\n")


def build_pipeline(args, device, rank):
    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "restrict_self_attn": False,
        "guidance_cross_attn": True,
        "is_train_restrict_lora": False,
        "restrict_lora": False,
        "restrict_lora_rank": 128,
        "is_amplify_history": False,
        "history_scale_mode": "per_head",
        "train_norm_layers": False,
        "is_control_model": True,
    }

    transformer = HeliosTransformer3DModelFunCtrl.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=args.weight_dtype,
        transformer_additional_kwargs=transformer_additional_kwargs,
    )

    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()

    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    if args.is_enable_stage2:
        scheduler = HeliosScheduler(
            shift=args.stage2_timestep_shift,
            stages=args.stage2_num_stages,
            stage_range=args.stage2_stage_range,
            gamma=args.stage2_scheduler_gamma,
        )
        pipe = HeliosFunCtrlPipeline.from_pretrained(
            args.base_model_path,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=args.weight_dtype,
        )
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
        )

        pipe = HeliosFunCtrlPipeline.from_pretrained(
            args.base_model_path,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=args.weight_dtype,
        )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.lora_path is not None:
        rank0_print(rank, "\n🎨 Loading LoRA Weights")
        rank0_print(rank, f"📁 LoRA path: {args.lora_path}")

        pipe.load_lora_weights(args.lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])

        rank0_print(rank, "✅ LoRA loaded successfully")

        if args.partial_path is not None:
            rank0_print(rank, "\n🧩 Loading Partial Components")
            rank0_print(rank, f"📁 Partial path: {args.partial_path}")

            if not hasattr(args, "training_config"):
                from argparse import Namespace
                args.training_config = Namespace()

            args.training_config.is_enable_stage1 = True
            args.training_config.restrict_self_attn = True
            args.training_config.is_amplify_history = True
            args.training_config.is_use_gan = True

            load_extra_components(args, transformer, args.partial_path)
            rank0_print(rank, "✅ Partial components loaded")

    print_model_info(transformer, pipe=pipe, rank=rank)

    if args.vae_decode_type == "once":
        pipe.vae.enable_tiling()

    transformer.eval()
    transformer.requires_grad_(False)
    vae.eval()
    vae.requires_grad_(False)

    transformer.to(device)
    vae.to(device)
    pipe.to(device)

    return pipe, transformer, vae


def select_ode_timesteps(stage_timesteps):
    target_timesteps = torch.tensor(
        [
            998.5342,
            902.2183,
            833.9636,
            783.0660,
            742.8216,
            640.0038,
            547.1926,
            462.9951,
            385.4137,
            328.6249,
            253.9905,
            151.5308,
        ],
        device=stage_timesteps.device,
        dtype=stage_timesteps.dtype,
    )

    save_timestep_indices = []
    seen = set()

    for target_t in target_timesteps:
        closest_idx = torch.argmin(torch.abs(stage_timesteps - target_t)).item()
        if closest_idx not in seen:
            save_timestep_indices.append(closest_idx)
            seen.add(closest_idx)

    save_latent_indices = save_timestep_indices + [-1]
    return save_timestep_indices, save_latent_indices


def process_one_item(batch, args, pipe, device, rank):
    assert len(batch["uttid"]) == 1

    pt_path = batch["pt_path"][0]
    output_path = batch["output_path"][0]

    if os.path.exists(output_path):
        print(f"⏭️ [rank={rank}] Skipping existing: {output_path}")
        return "skipped", output_path

    pt_item = torch.load(pt_path, map_location="cpu", weights_only=False)

    prompt_raw = pt_item["prompt_raw"]

    control_latents = pt_item["control_latent"].to(
        device=device,
        dtype=torch.float32,
    )

    gt_vae_latent = pt_item["vae_latent"]

    if gt_vae_latent.ndim == 4:
        gt_vae_latent = gt_vae_latent.unsqueeze(0)
    elif gt_vae_latent.ndim == 5:
        gt_vae_latent = gt_vae_latent[0].unsqueeze(0)

    _ = gt_vae_latent[:, :, :1].to(device=device, dtype=torch.float32)

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    with torch.no_grad():
        all_sections_ode = pipe(
            prompt=prompt_raw,
            control_latents=control_latents,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="latent",
            vae_decode_type=args.vae_decode_type,
            history_sizes=[16, 2, 1],
            latent_window_size=args.latent_window_size,
            is_keep_x0=True,
            use_dynamic_shifting=args.use_dynamic_shifting,
            time_shift_type=args.time_shift_type,
            use_cfg_zero_star=args.use_cfg_zero_star,
            use_zero_init=args.use_zero_init,
            zero_steps=args.zero_steps,
        )

    processed_sections_ode = []

    for section_ode in all_sections_ode:
        processed_section_ode = []

        for stage_ode in section_ode:
            stage_timesteps = stage_ode["timesteps"]
            save_timestep_indices, save_latent_indices = select_ode_timesteps(stage_timesteps)

            processed_stage_ode = {
                "latents": stage_ode["latents"][save_latent_indices],
                "timesteps": stage_timesteps[save_timestep_indices],
                "sigmas": stage_ode["sigmas"][save_timestep_indices].cpu(),
            }

            if "noise_pred" in stage_ode:
                processed_stage_ode["noise_pred"] = (
                    stage_ode["noise_pred"][save_timestep_indices].cpu()
                )

            if "control_latents" in stage_ode:
                processed_stage_ode["control_latents"] = (
                    stage_ode["control_latents"][save_latent_indices].cpu()
                )

            if "model_input" in stage_ode:
                processed_stage_ode["model_input"] = (
                    stage_ode["model_input"][save_timestep_indices].cpu()
                )

            processed_section_ode.append(processed_stage_ode)

        processed_sections_ode.append(processed_section_ode)

    temp_to_save = {
        "latent_window_size": args.latent_window_size,
        "prompt_raw": prompt_raw,
        "prompt_embed": pt_item["prompt_embed"].cpu(),
        "prompt_attention_mask": pt_item.get("prompt_attention_mask", None),
        "gt_vae_latent": pt_item["vae_latent"].cpu(),
        "control_latent": pt_item["control_latent"].cpu(),
        "ode_latents": processed_sections_ode,
        "source_pt": pt_path,
    }

    tmp_output_path = f"{output_path}.rank{rank}.tmp"
    torch.save(temp_to_save, tmp_output_path)
    os.replace(tmp_output_path, output_path)

    return "saved", output_path


def reduce_stats(device, saved, skipped, failed):
    stats = torch.tensor([saved, skipped, failed], device=device, dtype=torch.long)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    return stats


def main():
    args = parse_args()

    local_rank, rank, world_size = setup_distributed_env()
    device = torch.cuda.current_device()

    # accelerator = Accelerator()

    if args.feature_folder is not None:
        feature_folders = [args.feature_folder]
    else:
        feature_folders = [
            "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view/latents_short_giga_control",
        ]

    if args.output_folder is not None:
        output_folders = [args.output_folder]
    else:
        output_folders = [
            "/shared_disk/users/zhanqian.wu/data/train_data/debug/helios_data_stage3_FlowMatchEulerDiscreteScheduler",
        ]

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    rank0_print(rank, "\n🚀 ===== FunCtrl ODE Pair Generation =====")
    rank0_print(rank, f"🌍 world_size={world_size}")
    rank0_print(rank, f"🎮 local_rank={local_rank}, device=cuda:{local_rank}")
    rank0_print(rank, "========================================\n")

    pipe, transformer, vae = build_pipeline(args, device, rank)

    for feature_folder, output_folder in zip(feature_folders, output_folders):
        payload = prepare_pt_dataset_on_rank0(feature_folder, output_folder, rank)

        jobs = payload["jobs"]
        total = payload["total"]
        skipped_existing = payload["skipped"]
        todo = payload["todo"]
        broken = payload["broken"]

        if rank == 0:
            completion = 100.0 * skipped_existing / max(total, 1)

            print("\n" + "=" * 80)
            print("📂 Dataset Summary")
            print("=" * 80)
            print(f"📥 Input Folder  : {feature_folder}")
            print(f"📤 Output Folder : {output_folder}")
            print("-" * 80)
            print(f"📦 Total Files   : {total:,}")
            print(f"✅ Existing      : {skipped_existing:,}")
            print(f"🚧 To Process    : {todo:,}")
            print(f"⚠️ Broken Names  : {broken:,}")
            print(f"📈 Completion    : {completion:.2f}%")
            print("=" * 80)

        if todo == 0:
            rank0_print(rank, "🎉 All files already exist. Nothing to do.\n")
            continue

        dataset = PtLatentDataset(jobs)

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        ) if world_size > 1 else None

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False if sampler is not None else False,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
            pin_memory=True,
            drop_last=False,
        )

        # dataloader = accelerator.prepare(dataloader)

        rank0_print(rank, "\n🧩 Dataloader")
        rank0_print(rank, f"├── 🌍 world size : {world_size}")
        rank0_print(rank, f"├── 🚧 global todo: {todo:,}")
        rank0_print(rank, f"└── 📊 rank0 steps: {len(dataloader):,}\n")

        local_saved = 0
        local_skipped = 0
        local_failed = 0
        global_last_saved = 0
        start_time = time.time()

        pbar = tqdm(
            total=todo,
            initial=0,
            desc="🚀 Global ODE",
            position=0,
            dynamic_ncols=True,
            leave=True,
            disable=(rank != 0),
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        local_iter = tqdm(
            dataloader,
            desc=f"🧩 Rank {rank:02d}",
            position=rank + 1,
            dynamic_ncols=True,
            leave=False,
            colour="green",
            disable=(rank == 0),
        )

        for batch in local_iter:
            try:
                status, path = process_one_item(
                    batch=batch,
                    args=args,
                    pipe=pipe,
                    device=device,
                    rank=rank,
                )

                if status == "saved":
                    local_saved += 1
                elif status == "skipped":
                    local_skipped += 1

            except Exception as e:
                local_failed += 1
                print(f"\n❌ [rank={rank}] failed: {e}", flush=True)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            global_stats = reduce_stats(device, local_saved, local_skipped, local_failed)
            global_saved = int(global_stats[0].item())
            global_skipped = int(global_stats[1].item())
            global_failed = int(global_stats[2].item())

            if rank == 0:
                delta = global_saved - global_last_saved
                if delta > 0:
                    pbar.update(delta)
                    global_last_saved = global_saved

                done_total = skipped_existing + global_saved + global_skipped
                pct = 100.0 * done_total / max(total, 1)

                pbar.set_postfix_str(
                    f"✅ exists={skipped_existing:,} "
                    f"💾 saved={global_saved:,} "
                    f"⏭️ skip={global_skipped:,} "
                    f"❌ fail={global_failed:,} "
                    f"📈 all={pct:.2f}%"
                )

        if rank == 0:
            pbar.close()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        elapsed = time.time() - start_time
        final_stats = reduce_stats(device, local_saved, local_skipped, local_failed)

        if rank == 0:
            final_saved = int(final_stats[0].item())
            final_skipped = int(final_stats[1].item())
            final_failed = int(final_stats[2].item())

            print("\n" + "=" * 80)
            print("🎉 Generation Finished")
            print("=" * 80)
            print(f"📦 Total Files    : {total:,}")
            print(f"✅ Pre-existing   : {skipped_existing:,}")
            print(f"💾 Newly Saved    : {final_saved:,}")
            print(f"⏭️ Runtime Skipped: {final_skipped:,}")
            print(f"❌ Failed         : {final_failed:,}")
            print(f"⚠️ Broken Names   : {broken:,}")
            print(f"📈 Final Done     : {skipped_existing + final_saved + final_skipped:,}/{total:,}")
            print(f"⏱️ Elapsed        : {elapsed / 60:.2f} min")
            print("=" * 80 + "\n")

    cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FunCtrl ODE pairs with multi-GPU support")

    parser.add_argument("--base_model_path", type=str, default="/mnt/pfs/users/zhanqian.wu/ckpt/stage-1/stage1_final")
    parser.add_argument("--transformer_path", type=str, default="/mnt/pfs/users/zhanqian.wu/ckpt/stage-3-init/stage1_final_3v_uni_s16k")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--partial_path", type=str, default=None)
    parser.add_argument("--use_default_loader", action="store_true")

    parser.add_argument("--sample_type", type=str, default="t2v", choices=["t2v", "i2v", "v2v"])
    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--feature_folder", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default=None)

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--num_frames", type=int, default=165)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--use_dynamic_shifting", action="store_true")
    parser.add_argument("--time_shift_type", type=str, default="linear", choices=["exponential", "linear"])
    parser.add_argument("--vae_decode_type", type=str, default="default", choices=["default", "once", "default_fast"])

    parser.add_argument("--latent_window_size", type=int, default=9)

    parser.add_argument("--is_enable_stage2", action="store_true")
    parser.add_argument("--stage2_timestep_shift", type=float, default=1.0)
    parser.add_argument("--stage2_scheduler_gamma", type=float, default=1 / 3)
    parser.add_argument("--stage2_stage_range", type=float, nargs="+", default=[0, 1 / 3, 2 / 3, 1])
    parser.add_argument("--stage2_num_stages", type=int, default=3)
    parser.add_argument("--stage2_num_inference_steps_list", type=int, nargs="+", default=[20, 20, 20])

    parser.add_argument("--use_cfg_zero_star", action="store_true")
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )
    parser.add_argument("--prompt_txt_path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main()