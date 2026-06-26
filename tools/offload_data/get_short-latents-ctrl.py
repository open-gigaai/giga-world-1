import argparse
import os

import imageio
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from accelerate import Accelerator
from helios.dataset.dataloader_mp4_dist import (
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


def setup_distributed_env():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed_env():
    dist.destroy_process_group()


def video_to_sobel_edges(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    pixel_values: [B, 3, T, H, W], range [-1, 1]
    return:       [B, 3, T, H, W], range [-1, 1]
    """
    gray = (
        0.299 * pixel_values[:, 0:1]
        + 0.587 * pixel_values[:, 1:2]
        + 0.114 * pixel_values[:, 2:3]
    )

    b, _, t, h, w = gray.shape
    gray_2d = gray.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)

    sobel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        dtype=gray_2d.dtype,
        device=gray_2d.device,
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]],
        dtype=gray_2d.dtype,
        device=gray_2d.device,
    ).unsqueeze(0)

    grad_x = F.conv2d(gray_2d, sobel_x, padding=1)
    grad_y = F.conv2d(gray_2d, sobel_y, padding=1)

    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
    edge = edge * 2.0 - 1.0

    edge = edge.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4)
    edge = edge.repeat(1, 3, 1, 1, 1)

    return edge


def save_debug_video(video_tensor, save_path, fps=16):
    """
    video_tensor: [3, T, H, W], range [-1, 1]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    video = ((video_tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).cpu().numpy()

    writer = imageio.get_writer(save_path, fps=fps, codec="libx264")
    for frame in video:
        writer.append_data(frame)
    writer.close()


def save_sobel_debug_videos(
    pixel_values,
    control_pixel_values,
    debug_dir,
    batch_idx,
    rank,
    max_debug_batches=10,
    max_debug_samples=2,
    fps=16,
):
    """
    保存 原图 | Sobel 对比视频
    pixel_values/control_pixel_values: [B, 3, T, H, W]
    """
    if rank != 0:
        return

    if batch_idx >= max_debug_batches:
        return

    os.makedirs(debug_dir, exist_ok=True)

    num_samples = min(max_debug_samples, pixel_values.shape[0])

    for sample_idx in range(num_samples):
        orig = pixel_values[sample_idx]
        sobel = control_pixel_values[sample_idx]

        # [3, T, H, W * 2]
        concat_video = torch.cat([orig, sobel], dim=-1)

        save_path = os.path.join(
            debug_dir,
            f"batch{batch_idx:04d}_sample{sample_idx}_orig_sobel.mp4",
        )

        print(f"🎬 [Debug] Saving Sobel compare video: {save_path}")

        save_debug_video(
            video_tensor=concat_video,
            save_path=save_path,
            fps=fps,
        )


def encode_video_by_chunks(
    vae,
    pixel_values: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    latent_window_size: int = 9,
):
    """
    pixel_values: [B, 3, T, H, W]
    return: [B, num_chunks, 16, latent_T, H//8, W//8]
    """
    frame_window_size = (latent_window_size - 1) * 4 + 1
    num_frames = pixel_values.shape[2]
    num_chunk_to_encode = num_frames // frame_window_size

    latent_list = []

    for i in range(num_chunk_to_encode):
        start_idx = i * frame_window_size
        end_idx = start_idx + frame_window_size

        cur_pixel_values = pixel_values[:, :, start_idx:end_idx, :, :]

        cur_latent = vae.encode(cur_pixel_values).latent_dist.sample()
        cur_latent = (cur_latent - latents_mean) * latents_std

        latent_list.append(cur_latent)

    if len(latent_list) == 0:
        return None

    return torch.stack(latent_list, dim=1)


def main(
    rank,
    world_size,
    global_rank,
    stride,
    batch_size,
    dataloader_num_workers,
    json_file,
    video_folder,
    output_latent_folder,
    pretrained_model_name_or_path,
    resolution=640,
    debug_dir="./debug_sobel",
    max_debug_batches=10,
):
    weight_dtype = torch.bfloat16
    device = rank
    seed = 42

    print("🚀 [Init] Start preprocessing with Sobel control")
    print(f"   🔹 rank: {rank}")
    print(f"   🔹 world_size: {world_size}")
    print(f"   🔹 json_file: {json_file}")
    print(f"   🔹 video_folder: {video_folder}")
    print(f"   🔹 output_latent_folder: {output_latent_folder}")
    print(f"   🔹 debug_dir: {debug_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    latents_mean = torch.tensor(
        vae.config.latents_mean
    ).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)

    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std
    ).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)

    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    dataset = BucketedFeatureDataset(
        json_files=json_file,
        video_folders=video_folder,
        stride=stride,
        force_rebuild=True,
        resolution=resolution,
        single_res=True,
        single_height=384,
        single_width=640,
    )

    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
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

    print(f"✅ Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")
    print(f"✅ Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")

    sampler.set_epoch(0)

    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="[INFO 🎬] Processing")

    os.makedirs(output_latent_folder, exist_ok=True)

    for idx, batch in enumerate(dataloader):
        if batch is None or batch["videos"] is None:
            print("⚠️ None batch, continuing")
            continue

        free_memory()

        valid_uttids = []
        valid_num_frames = []
        valid_heights = []
        valid_widths = []
        valid_videos = []
        valid_prompts = []
        valid_first_frames_images = []

        if batch["uttid"] is None:
            print("⚠️ None uttid batch, continuing")
            continue

        for i, (uttid, num_frame, height, width) in enumerate(
            zip(
                batch["uttid"],
                batch["video_metadata"]["num_frames"],
                batch["video_metadata"]["height"],
                batch["video_metadata"]["width"],
            )
        ):
            output_path = os.path.join(
                output_latent_folder,
                f"{uttid}_{num_frame}_{height}_{width}.pt",
            )

            if not os.path.exists(output_path):
                valid_uttids.append(uttid)
                valid_num_frames.append(num_frame)
                valid_heights.append(height)
                valid_widths.append(width)
                valid_videos.append(batch["videos"][i])
                valid_prompts.append(batch["prompts"][i])
                valid_first_frames_images.append(batch["first_frames_images"][i])
            else:
                print(f"⏭️ skipping existing: {uttid}")

        if not valid_uttids:
            print("⏭️ skipping entire batch!")
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({"batch": idx})
            continue

        batch = {
            "uttid": valid_uttids,
            "video_metadata": {
                "num_frames": valid_num_frames,
                "height": valid_heights,
                "width": valid_widths,
            },
            "videos": torch.stack(valid_videos),
            "prompts": valid_prompts,
            "first_frames_images": torch.stack(valid_first_frames_images),
        }

        with torch.no_grad():
            # [B, T, C, H, W] -> [B, C, T, H, W]
            pixel_values = batch["videos"].permute(0, 2, 1, 3, 4).to(
                dtype=vae.dtype,
                device=device,
            )

            print(f"🎬 [Batch {idx}] pixel_values: {tuple(pixel_values.shape)}")

            control_pixel_values = video_to_sobel_edges(pixel_values)

            print(f"🎮 [Batch {idx}] control_pixel_values: {tuple(control_pixel_values.shape)}")

            save_sobel_debug_videos(
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
                print("⚠️ No valid video chunks, skipping")
                continue

            control_latents = encode_video_by_chunks(
                vae=vae,
                pixel_values=control_pixel_values,
                latents_mean=latents_mean,
                latents_std=latents_std,
                latent_window_size=9,
            )

            if control_latents is None:
                print("⚠️ No valid control chunks, skipping")
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
            output_path = os.path.join(
                output_latent_folder,
                f"{uttid}_{num_frame}_{height}_{width}.pt",
            )

            temp_to_save = {
                "vae_latent": cur_vae_latent.cpu().detach(),
                "control_latent": cur_control_latent.cpu().detach(),
                "prompt_embed": cur_prompt_embed.cpu().detach(),
                "first_frames_image": cur_first_frames_image,
                "prompt_raw": cur_prompt,
                "control_type": "sobel_edge",
            }

            try:
                torch.save(temp_to_save, output_path)
                print(f"✅ save latent to: {output_path}")
            except Exception as e:
                print(f"❌ failed saving {output_path}: {e}")
                continue

        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({"batch": idx})

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

    print("🎉 [Done] Sobel control latent preprocessing finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Wan latents with Sobel edge control latents."
    )

    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/pfs/users/zhanqian.wu/ckpt/Wan2.1-T2V-1.3B-Diffusers",
    )

    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./debug_sobel",
    )

    parser.add_argument(
        "--max_debug_batches",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    setup_distributed_env()

    global_rank = dist.get_rank()
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()

    base_video_path = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/agibot_debug/"
    video_paths = ["videos"]

    base_csv_paths = [
        "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/agibot_debug/",
    ]
    csv_paths = ["helios_front_dataset.json"]

    base_output_latent_path = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/agibot_debug"
    output_latent_paths = ["latents_short_sobel_control"]

    resolutions = [640]
    strides = [1]
    batch_sizes = [4]

    for (
        stride,
        batch_size,
        base_csv_path,
        csv_path,
        video_path,
        output_latent_path,
        cur_resolution,
    ) in zip(
        strides,
        batch_sizes,
        base_csv_paths,
        csv_paths,
        video_paths,
        output_latent_paths,
        resolutions,
    ):
        json_file = os.path.join(base_csv_path, csv_path)
        video_folder = os.path.join(base_video_path, video_path)
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
            output_latent_folder=output_latent_folder,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            resolution=cur_resolution,
            debug_dir=args.debug_dir,
            max_debug_batches=args.max_debug_batches,
        )

    dist.barrier()
    cleanup_distributed_env()