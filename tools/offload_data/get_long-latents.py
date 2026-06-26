import argparse
import os

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from accelerate import Accelerator
from helios.dataset.dataloader_mp4_dist import BucketedFeatureDataset, BucketedSampler, collate_fn
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
):
    weight_dtype = torch.bfloat16
    device = rank
    seed = 42

    # Load the tokenizers
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

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, weight_dtype
    )

    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    # dist.barrier()
    dataset = BucketedFeatureDataset(
        json_files=json_file,
        video_folders=video_folder,
        stride=stride,
        force_rebuild=False,
        resolution=resolution,
        single_res=True,
        single_height=384,
        single_width=640,
        single_length=True,
        single_num_frame=81,
    )
    sampler = BucketedSampler(dataset, batch_size=batch_size, drop_last=False, shuffle=True, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2 if dataloader_num_workers != 0 else None,
        # persistent_workers=True if dataloader_num_workers > 0 else False,
    )

    print(len(dataset), len(dataloader))
    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")
    print(f"Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")

    sampler.set_epoch(0)
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Processing")
    # dist.barrier()
    for idx, batch in enumerate(dataloader):
        if batch is None or batch["videos"] is None:
            print("None batch, continuing")
            continue
        free_memory()

        valid_indices = []
        valid_uttids = []
        valid_num_frames = []
        valid_heights = []
        valid_widths = []
        valid_videos = []
        valid_prompts = []
        valid_first_frames_images = []

        if batch["uttid"] is None:
            print("None batch, contiuning")
            continue

        for i, (uttid, num_frame, height, width) in enumerate(
            zip(
                batch["uttid"],
                batch["video_metadata"]["num_frames"],
                batch["video_metadata"]["height"],
                batch["video_metadata"]["width"],
            )
        ):
            os.makedirs(output_latent_folder, exist_ok=True)
            output_path = os.path.join(output_latent_folder, f"{uttid}_{num_frame}_{height}_{width}.pt")
            if not os.path.exists(output_path):
                valid_indices.append(i)
                valid_uttids.append(uttid)
                valid_num_frames.append(num_frame)
                valid_heights.append(height)
                valid_widths.append(width)
                valid_videos.append(batch["videos"][i])
                valid_prompts.append(batch["prompts"][i])
                valid_first_frames_images.append(batch["first_frames_images"][i])
            else:
                print(f"skipping {uttid}")

        if not valid_indices:
            print("skipping entire batch!")
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({"batch": idx})
            continue

        batch = None
        del batch
        free_memory()

        batch = {
            "uttid": valid_uttids,
            "video_metadata": {"num_frames": valid_num_frames, "height": valid_heights, "width": valid_widths},
            "videos": torch.stack(valid_videos),
            "prompts": valid_prompts,
            "first_frames_images": torch.stack(valid_first_frames_images),
        }

        if len(batch["uttid"]) == 0:
            print("All samples in this batch are already processed, skipping!")
            continue

        with torch.no_grad():
            # Get Vae feature
            pixel_values = batch["videos"].permute(0, 2, 1, 3, 4).to(dtype=vae.dtype, device=device)
            vae_latents = vae.encode(pixel_values).latent_dist.sample()
            vae_latents = (vae_latents - latents_mean) * latents_std

            # Encode prompts
            prompts = batch["prompts"]
            prompt_embeds, prompt_attention_mask = encode_prompt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=prompts,
                device=device,
            )

            image_tensor = batch["first_frames_images"]
            images = [transforms.ToPILImage()(x.to(torch.uint8)) for x in image_tensor]

        for (
            uttid,
            num_frame,
            height,
            width,
            cur_vae_latent,
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
            prompt_embeds,
            prompt_attention_mask,
            images,
            prompts,
        ):
            output_path = os.path.join(output_latent_folder, f"{uttid}_{num_frame}_{height}_{width}.pt")
            temp_to_save = {
                "vae_latent": cur_vae_latent.cpu().detach(),
                "prompt_embed": cur_prompt_embed.cpu().detach(),
                # "prompt_attention_mask": cur_prompt_attention_mask.cpu().detach(),
                "first_frames_image": cur_first_frames_image,
                "prompt_raw": cur_prompt,
            }
            try:
                torch.save(temp_to_save, output_path)
            except Exception:
                continue
            print(f"save latent to: {output_path}")

        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({"batch": idx})

        pixel_values = None
        prompts = None
        image_tensor = None
        images = None
        vae_latents = None
        vae_latents_2 = None
        image_embeds = None
        prompt_embeds = None
        batch = None
        valid_indices = None
        valid_uttids = None
        valid_num_frames = None
        valid_heights = None
        valid_widths = None
        valid_videos = None
        valid_prompts = None
        valid_first_frames_images = None
        temp_to_save = None

        del pixel_values
        del prompts
        del image_tensor
        del images
        del vae_latents
        del vae_latents_2
        del image_embeds
        del batch
        del valid_indices
        del valid_uttids
        del valid_num_frames
        del valid_heights
        del valid_widths
        del valid_videos
        del valid_prompts
        del valid_first_frames_images
        del temp_to_save

        free_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running model training and data processing.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="BestWishYsh/Helios-Base",
        help="Pretrained model path",
    )
    args = parser.parse_args()

    setup_distributed_env()

    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()

    base_video_path = "example"
    video_paths = [
        "toy_data",
    ]

    base_output_latent_path = "example/toy_data/latents_long"
    output_latent_paths = [
        "toy_data",
    ]

    base_csv_paths = [
        "example",
    ]
    csv_paths = [
        "toy_data/toy_filter.json",
    ]

    resolutions = [640]
    strides = [1]
    batch_sizes = [4]

    for stride, batch_size, base_csv_path, csv_path, video_path, output_latent_path, cur_resolution in zip(
        strides, batch_sizes, base_csv_paths, csv_paths, video_paths, output_latent_paths, resolutions
    ):
        json_file = os.path.join(base_csv_path, csv_path)
        video_folder = os.path.join(base_video_path, video_path)
        output_latent_folder = os.path.join(base_output_latent_path, output_latent_path)

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
        )

    dist.barrier()
    dist.destroy_process_group()
