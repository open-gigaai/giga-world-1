import argparse
import os
import time

import imageio
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from accelerate import Accelerator
from helios.dataset.dataloader_mp4_dist_single_view import (
    BucketedFeatureDataset,
    BucketedSampler,
    collate_fn,
)
from helios.utils.utils_base import encode_prompt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan
from diffusers.training_utils import free_memory

MIN_VALID_PT_SIZE = 1024


def setup_distributed_env():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed_env():
    dist.destroy_process_group()


def is_valid_pt(path, min_size=MIN_VALID_PT_SIZE):
    return os.path.exists(path) and os.path.getsize(path) >= min_size


def atomic_torch_save(obj, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    torch.save(obj, tmp_path)
    os.replace(tmp_path, output_path)


def save_debug_video(video_tensor, save_path, fps=16):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video = ((video_tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).cpu().numpy()

    writer = imageio.get_writer(save_path, fps=fps, codec="libx264")
    for frame in video:
        writer.append_data(frame)
    writer.close()


def save_control_debug_videos(
    pixel_values,
    control_pixel_values,
    debug_dir,
    batch_idx,
    rank,
    max_debug_batches=1,
    max_debug_samples=1,
    fps=16,
):
    if rank != 0:
        return
    if batch_idx >= max_debug_batches:
        return

    os.makedirs(debug_dir, exist_ok=True)
    num_samples = min(max_debug_samples, pixel_values.shape[0])

    for sample_idx in range(num_samples):
        rgb = pixel_values[sample_idx]
        ctrl = control_pixel_values[sample_idx]
        concat_video = torch.cat([rgb, ctrl], dim=-1)

        save_path = os.path.join(
            debug_dir,
            f"batch{batch_idx:04d}_sample{sample_idx}_rgb_control.mp4",
        )

        print(f"🎬 [Debug] Saving RGB-Control compare video: {save_path}")
        save_debug_video(concat_video, save_path, fps=fps)


def encode_video_by_chunks(
    vae,
    pixel_values,
    latents_mean,
    latents_std,
    latent_window_size=9,
    offload_to_cpu=True,
):
    frame_window_size = (latent_window_size - 1) * 4 + 1
    num_frames = pixel_values.shape[2]
    num_chunk_to_encode = num_frames // frame_window_size

    if num_chunk_to_encode <= 0:
        return None

    latent_list = []

    for i in range(num_chunk_to_encode):
        start_idx = i * frame_window_size
        end_idx = start_idx + frame_window_size

        cur_pixel_values = pixel_values[:, :, start_idx:end_idx].contiguous()

        with torch.no_grad():
            cur_latent = vae.encode(cur_pixel_values).latent_dist.sample()

        cur_latent = cur_latent.detach()
        cur_latent = (cur_latent - latents_mean) * latents_std

        if offload_to_cpu:
            cur_latent = cur_latent.cpu()

        latent_list.append(cur_latent)

        del cur_pixel_values
        torch.cuda.empty_cache()

    return torch.stack(latent_list, dim=1)


def rebuild_buckets_after_filter(dataset):
    from collections import defaultdict

    new_buckets = defaultdict(list)

    for new_idx, sample in enumerate(dataset.samples):
        bucket_key = sample["bucket_key"]
        new_buckets[bucket_key].append(new_idx)

    dataset.buckets = new_buckets
    return dataset


def get_output_path(output_latent_folder, uttid, num_frame, height, width):
    return os.path.join(
        output_latent_folder,
        f"{uttid}_{num_frame}_{height}_{width}.pt",
    )


def filter_dataset_append_mode(dataset, output_latent_folder, append_mode=True):
    if not append_mode:
        print("🧱 [Append Mode] disabled, process all samples")
        return dataset

    if not hasattr(dataset, "samples"):
        print("⚠️ dataset has no attribute `samples`, skip append filtering")
        return dataset

    old_len = len(dataset.samples)
    filtered_samples = []

    print("🔍 [Append Mode] scanning existing pt files...")

    for item in tqdm(dataset.samples):
        try:
            uttid = item["uttid"]
            num_frame = item["bucket_num_frame"]
            height = item["bucket_height"]
            width = item["bucket_width"]

            output_path = get_output_path(
                output_latent_folder,
                uttid,
                num_frame,
                height,
                width,
            )

            if not is_valid_pt(output_path):
                filtered_samples.append(item)

        except Exception as e:
            print(f"⚠️ filter failed, keep item: {e}")
            filtered_samples.append(item)

    dataset.samples = filtered_samples
    dataset = rebuild_buckets_after_filter(dataset)

    print(
        f"✅ [Append Mode] "
        f"{old_len} -> {len(dataset.samples)} "
        f"(skip {old_len - len(dataset.samples)})"
    )

    return dataset


def build_dataset_with_rank0_cache(
    json_file,
    video_folder,
    control_video_folder,
    stride,
    resolution,
    global_rank,
    world_size,
):
    dataset_kwargs = dict(
        json_files=json_file,
        video_folders=video_folder,
        control_video_folders=control_video_folder,
        stride=stride,
        resolution=resolution,
        single_res=True,
        single_height=480,
        single_width=640,
    )

    if world_size > 1:
        cache_ready_path = os.path.join(
            os.path.dirname(json_file),
            f".{os.path.basename(json_file)}.{resolution}.stride{stride}.cache_ready",
        )

        if global_rank == 0:
            if os.path.exists(cache_ready_path):
                os.remove(cache_ready_path)

            print("🧱 [Cache] rank0 force rebuild dataset cache...")
            _ = BucketedFeatureDataset(
                **dataset_kwargs,
                force_rebuild=True,
            )
            with open(cache_ready_path, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
            print("✅ [Cache] rank0 rebuild done")
        else:
            wait_start_time = time.time()
            while not os.path.exists(cache_ready_path):
                waited = int(time.time() - wait_start_time)
                print(
                    f"⏳ [Cache] rank{global_rank} waiting rank0 cache "
                    f"ready... {waited}s",
                    flush=True,
                )
                time.sleep(60)

        print(f"📦 [Cache] rank{global_rank} load dataset cache...")
        dataset = BucketedFeatureDataset(
            **dataset_kwargs,
            force_rebuild=False,
        )
        return dataset

    print("🧱 [Cache] single process force rebuild dataset cache...")
    dataset = BucketedFeatureDataset(
        **dataset_kwargs,
        force_rebuild=True,
    )
    return dataset


def main(
    rank,
    world_size,
    global_rank,
    stride,
    batch_size,
    dataloader_num_workers,
    json_file,
    video_folder,
    control_video_folder,
    output_latent_folder,
    pretrained_model_name_or_path,
    resolution="giga_ctrl",
    debug_dir="./debug_control",
    max_debug_batches=10,
    append_mode=True,
    order_mode="sequential",
):
    weight_dtype = torch.bfloat16
    device = rank
    seed = 42

    assert order_mode in ["sequential", "random"]
    shuffle = order_mode == "random"

    print("🚀 [Init] Start preprocessing with external control video")
    print(f"   🔹 rank: {rank}")
    print(f"   🔹 world_size: {world_size}")
    print(f"   🔹 global_rank: {global_rank}")
    print(f"   🔹 json_file: {json_file}")
    print(f"   🔹 video_folder: {video_folder}")
    print(f"   🔹 control_video_folder: {control_video_folder}")
    print(f"   🔹 output_latent_folder: {output_latent_folder}")
    print(f"   🔹 append_mode: {append_mode}")
    print(f"   🔹 order_mode: {order_mode}")
    print(f"   🔹 shuffle: {shuffle}")

    if not os.path.isdir(pretrained_model_name_or_path):
        raise FileNotFoundError(pretrained_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        local_files_only=True,
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )

    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )

    vae.enable_tiling()
    vae.tile_sample_min_height = 256
    vae.tile_sample_min_width = 320
    vae.tile_overlap_factor_height = 0.25
    vae.tile_overlap_factor_width = 0.25

    latents_mean = torch.tensor(
        vae.config.latents_mean
    ).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)

    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std
    ).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)

    vae.eval().requires_grad_(False)
    text_encoder.eval().requires_grad_(False)

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    dataset = build_dataset_with_rank0_cache(
        json_file=json_file,
        video_folder=video_folder,
        control_video_folder=control_video_folder,
        stride=stride,
        resolution=resolution,
        global_rank=global_rank,
        world_size=world_size,
    )

    dataset = filter_dataset_append_mode(
        dataset=dataset,
        output_latent_folder=output_latent_folder,
        append_mode=append_mode,
    )

    if len(dataset) == 0:
        print("🎉 all samples already processed.")
        return

    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=shuffle,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2 if dataloader_num_workers != 0 else None,
    )

    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)

    print(
        f"✅ Dataset size: {len(dataset)}, "
        f"Dataloader batches: {len(dataloader)}"
    )

    print(
        f"✅ Process index: {accelerator.process_index}, "
        f"World size: {accelerator.num_processes}"
    )

    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="[INFO 🎬] Processing")
        progress_start_time = time.time()
        processed_samples = 0

    os.makedirs(output_latent_folder, exist_ok=True)

    for idx, batch in enumerate(dataloader):
        if batch is None:
            print("⚠️ None batch")
            continue

        if batch.get("videos") is None:
            print("⚠️ batch videos is None")
            continue

        if "control_videos" not in batch:
            raise KeyError("batch missing `control_videos`")

        free_memory()

        valid_uttids = []
        valid_num_frames = []
        valid_heights = []
        valid_widths = []
        valid_videos = []
        valid_control_videos = []
        valid_prompts = []
        valid_first_frames_images = []

        if batch["uttid"] is None:
            print("⚠️ None uttid batch")
            continue

        for i, (uttid, num_frame, height, width) in enumerate(
            zip(
                batch["uttid"],
                batch["video_metadata"]["num_frames"],
                batch["video_metadata"]["height"],
                batch["video_metadata"]["width"],
            )
        ):
            output_path = get_output_path(
                output_latent_folder,
                uttid,
                num_frame,
                height,
                width,
            )

            if append_mode and is_valid_pt(output_path):
                print(f"⏭️ skip existing: {uttid}")
                continue

            valid_uttids.append(uttid)
            valid_num_frames.append(num_frame)
            valid_heights.append(height)
            valid_widths.append(width)
            valid_videos.append(batch["videos"][i])
            valid_control_videos.append(batch["control_videos"][i])
            valid_prompts.append(batch["prompts"][i])
            valid_first_frames_images.append(batch["first_frames_images"][i])

        if len(valid_uttids) == 0:
            print("⏭️ skipping entire batch")
            if rank == 0:
                pbar.update(1)
                elapsed = max(time.time() - progress_start_time, 1e-6)
                pbar.set_postfix(
                    {
                        "batch": idx,
                        "samples": processed_samples,
                        "batch/s": f"{pbar.n / elapsed:.2f}",
                        "samples/s": f"{processed_samples / elapsed:.2f}",
                    }
                )
            continue

        batch = {
            "uttid": valid_uttids,
            "video_metadata": {
                "num_frames": valid_num_frames,
                "height": valid_heights,
                "width": valid_widths,
            },
            "videos": torch.stack(valid_videos),
            "control_videos": torch.stack(valid_control_videos),
            "prompts": valid_prompts,
            "first_frames_images": torch.stack(valid_first_frames_images),
        }

        with torch.no_grad():
            pixel_values = batch["videos"].permute(0, 2, 1, 3, 4).to(
                dtype=vae.dtype,
                device=device,
            )

            control_pixel_values = batch["control_videos"].permute(
                0, 2, 1, 3, 4
            ).to(
                dtype=vae.dtype,
                device=device,
            )

            print(f"🎬 [Batch {idx}] pixel_values: {tuple(pixel_values.shape)}")
            print(
                f"🎮 [Batch {idx}] "
                f"control_pixel_values: {tuple(control_pixel_values.shape)}"
            )

            save_control_debug_videos(
                pixel_values=pixel_values,
                control_pixel_values=control_pixel_values,
                debug_dir=debug_dir,
                batch_idx=idx,
                rank=rank,
                max_debug_batches=max_debug_batches,
                max_debug_samples=2,
                fps=16,
            )

            vae_latents = encode_video_by_chunks(
                vae=vae,
                pixel_values=pixel_values,
                latents_mean=latents_mean,
                latents_std=latents_std,
                latent_window_size=9,
            )

            if vae_latents is None:
                print("⚠️ No valid video chunks")
                continue

            control_latents = encode_video_by_chunks(
                vae=vae,
                pixel_values=control_pixel_values,
                latents_mean=latents_mean,
                latents_std=latents_std,
                latent_window_size=9,
            )

            if control_latents is None:
                print("⚠️ No valid control chunks")
                continue

            print(f"✅ vae_latents: {tuple(vae_latents.shape)}")
            print(f"✅ control_latents: {tuple(control_latents.shape)}")

            prompts = batch["prompts"]

            prompt_embeds, prompt_attention_mask = encode_prompt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=prompts,
                device=device,
            )

            image_tensor = batch["first_frames_images"]
            images = [
                transforms.ToPILImage()(x.to(torch.uint8))
                for x in image_tensor
            ]

        for (
            uttid,
            num_frame,
            height,
            width,
            cur_vae_latent,
            cur_control_latent,
            cur_prompt_embed,
            cur_prompt_attention_mask,
            cur_first_frames_image,
            cur_prompt,
        ) in zip(
            batch["uttid"],
            batch["video_metadata"]["num_frames"],
            batch["video_metadata"]["height"],
            batch["video_metadata"]["width"],
            vae_latents,
            control_latents,
            prompt_embeds,
            prompt_attention_mask,
            images,
            prompts,
        ):
            output_path = get_output_path(
                output_latent_folder,
                uttid,
                num_frame,
                height,
                width,
            )

            if append_mode and is_valid_pt(output_path):
                print(f"⏭️ skip existing before save: {uttid}")
                continue

            temp_to_save = {
                "vae_latent": cur_vae_latent.cpu().detach(),
                "control_latent": cur_control_latent.cpu().detach(),
                "prompt_embed": cur_prompt_embed.cpu().detach(),
                "prompt_attention_mask": cur_prompt_attention_mask.cpu().detach(),
                "first_frames_image": cur_first_frames_image,
                "prompt_raw": cur_prompt,
                "control_type": "external_control_video",
            }

            try:
                atomic_torch_save(temp_to_save, output_path)
                print(f"✅ save latent to: {output_path}")
            except Exception as e:
                print(f"❌ failed saving {output_path}: {e}")
                continue

        if rank == 0:
            processed_samples += len(valid_uttids)
            elapsed = max(time.time() - progress_start_time, 1e-6)
            pbar.update(1)
            pbar.set_postfix(
                {
                    "batch": idx,
                    "samples": processed_samples,
                    "batch/s": f"{pbar.n / elapsed:.2f}",
                    "samples/s": f"{processed_samples / elapsed:.2f}",
                }
            )

        del pixel_values
        del control_pixel_values
        del vae_latents
        del control_latents
        del prompt_embeds
        del prompt_attention_mask
        del image_tensor
        del images
        del batch

        free_memory()

    if rank == 0:
        pbar.close()

    print("🎉 [Done] Control latent preprocessing finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Wan latents with external control video latents."
    )

    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/shared_disk/users/zhanqian.wu/model/Wan2.2-Fun-5B-Control-diffusers",
    )

    parser.add_argument("--debug_dir", type=str, default="./debug_control")
    parser.add_argument("--max_debug_batches", type=int, default=10)

    parser.add_argument(
        "--append_mode",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no_append_mode",
        action="store_false",
        dest="append_mode",
    )

    parser.add_argument(
        "--order_mode",
        type=str,
        default="sequential",
        choices=["sequential", "random"],
    )

    args = parser.parse_args()

    setup_distributed_env()

    global_rank = dist.get_rank()
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()

    base_video_path = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view"

    video_paths = ["videos"]
    control_video_paths = ["control_videos"]

    base_csv_paths = [
        "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view",
    ]

    csv_paths = [
        "helios_giga_ctrl.jsonl",
    ]

    base_output_latent_path = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_singleview_wan22_5b"

    output_latent_paths = [
        "latents_short_giga_control_single_view_wan22_5b",
    ]

    # base_video_path = "/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl"

    # video_paths = ["videos"]
    # control_video_paths = ["control_videos"]

    # base_csv_paths = [
    #     "/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl",
    # ]

    # csv_paths = [
    #     "helios_giga_ctrl.jsonl",
    # ]

    # base_output_latent_path = "/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl"

    # output_latent_paths = [
    #     "latents_short_giga_control_single_view_wan22_5b",
    # ]


    resolutions = ["giga_ctrl"]
    strides = [1]
    batch_sizes = [1]

    for (
        stride,
        batch_size,
        base_csv_path,
        csv_path,
        video_path,
        control_video_path,
        output_latent_path,
        cur_resolution,
    ) in zip(
        strides,
        batch_sizes,
        base_csv_paths,
        csv_paths,
        video_paths,
        control_video_paths,
        output_latent_paths,
        resolutions,
    ):
        json_file = os.path.join(base_csv_path, csv_path)
        video_folder = os.path.join(base_video_path, video_path)
        control_video_folder = os.path.join(base_video_path, control_video_path)
        output_latent_folder = os.path.join(
            base_output_latent_path,
            output_latent_path,
        )

        main(
            rank=device,
            world_size=world_size,
            global_rank=global_rank,
            stride=stride,
            batch_size=batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            json_file=json_file,
            video_folder=video_folder,
            control_video_folder=control_video_folder,
            output_latent_folder=output_latent_folder,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            resolution=cur_resolution,
            debug_dir=args.debug_dir,
            max_debug_batches=args.max_debug_batches,
            append_mode=args.append_mode,
            order_mode=args.order_mode,
        )

    dist.barrier()
    cleanup_distributed_env()