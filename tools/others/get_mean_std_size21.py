from datetime import datetime

import numpy as np
from helios.dataset.dataloader_dmd import BucketedFeatureDataset, BucketedSampler, collate_fn
from tqdm import tqdm


if __name__ == "__main__":
    # from diffusers import AutoencoderKLWan
    # vae = AutoencoderKLWan.from_pretrained(
    #     "BestWishYsh/Helios-Base",
    #     subfolder="vae",
    #     weight_dtype=torch.bfloat16,
    #     device_map="cuda",
    # )
    # vae.requires_grad_(False)
    # vae.eval()
    # from diffusers.utils import export_to_video
    # from diffusers.video_processor import VideoProcessor
    # vae_scale_factor_spatial = vae.spatial_compression_ratio
    # video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    # latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, dtype=vae.dtype))
    # latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, dtype=vae.dtype)

    from accelerate import Accelerator
    from torchdata.stateful_dataloader import StatefulDataLoader

    dataloader_num_workers = 96
    batch_size = 32
    num_train_epochs = 1
    seed = 0

    gan_folder = [
        "/mnt/hdfs/data/ysh_new/userful_things_wan/gan_latents/ultravideo/clips_long_960",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/gan_latents/ultravideo/clips_short_960",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/gan_latents/osp-sucai",
    ]
    accelerator = Accelerator()
    print(accelerator.process_index, accelerator.num_processes)

    dataset = BucketedFeatureDataset(
        gan_folders=gan_folder,
        force_rebuild=True,
        seed=seed,
    )
    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
        seed=seed,
        num_sp_groups=accelerator.num_processes // 1,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )
    dataloader = StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        prefetch_factor=2 if dataloader_num_workers > 0 else None,
    )
    print(len(dataset), len(dataloader))
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")

    max_samples = 500000
    output_txt = "latent_statistics_size21.txt"
    stats = {
        "x0_latents": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "target_latents": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
    }
    sampled_count = 0
    print(f"Starting to collect statistics from {max_samples} randomly sampled videos...")

    if accelerator.is_main_process:
        pbar = tqdm(total=max_samples, desc="Collecting statistics", unit="videos")

    step = 0
    global_step = 0
    first_epoch = 0
    print("Testing dataloader...")
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        for i, batch in enumerate(dataloader):
            if accelerator.is_main_process:
                x0_latents = batch["gan_vae_latents"][:, :, :1].cpu().float()
                target_latents = batch["gan_vae_latents"].cpu().float()

                stats["x0_latents"]["sum"] += x0_latents.sum().item()
                stats["x0_latents"]["sum_sq"] += (x0_latents**2).sum().item()
                stats["x0_latents"]["count"] += x0_latents.numel()

                stats["target_latents"]["sum"] += target_latents.sum().item()
                stats["target_latents"]["sum_sq"] += (target_latents**2).sum().item()
                stats["target_latents"]["count"] += target_latents.numel()

                sampled_count += x0_latents.shape[0]

                pbar.update(batch_size)

                if sampled_count % 1000 == 0:
                    print(f"Sampled {sampled_count}/{max_samples} videos...")

                if sampled_count >= max_samples:
                    break

        if sampled_count >= max_samples:
            break

    # 计算并保存统计结果
    if accelerator.is_main_process:
        # 准备输出内容
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("VAE Latent Statistics Report")
        output_lines.append("=" * 80)
        output_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append(f"Total sampled videos: {sampled_count}")
        output_lines.append(f"Random seed: {seed}")
        output_lines.append("=" * 80)
        output_lines.append("")

        # 计算每个latent的统计量
        for latent_name, stat in stats.items():
            if stat["count"] > 0:
                mean = stat["sum"] / stat["count"]
                variance = (stat["sum_sq"] / stat["count"]) - (mean**2)
                std = np.sqrt(variance)

                output_lines.append(f"{latent_name}:")
                output_lines.append(f"  Total elements: {stat['count']:,}")
                output_lines.append(f"  Mean (μ): {mean:.8f}")
                output_lines.append(f"  Variance (σ²): {variance:.8f}")
                output_lines.append(f"  Std Dev (σ): {std:.8f}")
                output_lines.append("")

        output_lines.append("=" * 80)
        output_lines.append("Recommended regularization values for training:")
        output_lines.append("=" * 80)

        # 计算平均值作为推荐参数
        mean_avg = np.mean([stats[k]["sum"] / stats[k]["count"] for k in stats.keys()])
        var_avg = np.mean(
            [
                (stats[k]["sum_sq"] / stats[k]["count"]) - (stats[k]["sum"] / stats[k]["count"]) ** 2
                for k in stats.keys()
            ]
        )

        output_lines.append(f"μ_target = {mean_avg:.8f}")
        output_lines.append(f"σ²_target = {var_avg:.8f}")
        output_lines.append("=" * 80)

        # 保存到文件
        result_text = "\n".join(output_lines)

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(result_text)

        # 同时打印到控制台
        print("\n" + result_text)
        print(f"\n✅ Statistics saved to: {output_txt}")

    print("\nStatistics collection completed!")
