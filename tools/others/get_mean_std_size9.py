import os
import pickle
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from einops import rearrange
from helios.dataset.dataloader_history_latents_dist import BucketedSampler, collate_fn
from torch.utils.data import Dataset
from tqdm import tqdm


class BucketedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_folders,
        history_sizes=[16, 2, 1],
        is_keep_x0=True,
        force_rebuild=False,
        return_all_vae_latent=False,
        return_prompt_raw=False,
        num_rollout_sections=3,
        single_res=False,
        single_height=384,
        single_width=640,
        seed=42,
    ):
        self.history_sizes = history_sizes
        self.is_keep_x0 = is_keep_x0
        self.force_rebuild = force_rebuild
        self.return_all_vae_latent = return_all_vae_latent
        self.return_prompt_raw = return_prompt_raw
        self.num_rollout_sections = num_rollout_sections
        self.single_res = single_res
        self.single_height = single_height
        self.single_width = single_width
        assert self.is_keep_x0, "is_keep_x0 need to be True now!"

        self.base_seed = seed
        self._epoch = 0

        if isinstance(feature_folders, str):
            self.feature_folders = [feature_folders]
        else:
            self.feature_folders = feature_folders

        self.samples = []
        self.buckets = defaultdict(list)

        for folder in self.feature_folders:
            cache_file = os.path.join(folder, "dataset_cache.pkl")
            self._process_folder(folder, cache_file)

    def _process_folder(self, folder, cache_file):
        if self.force_rebuild or not os.path.exists(cache_file):
            if os.path.exists(cache_file):
                os.remove(cache_file)
            print(f"Building metadata cache for folder: {folder}")
            folder_samples, folder_buckets = self._build_folder_metadata(folder)

            if not self.force_rebuild:
                print(f"Saving metadata cache for folder: {folder}")
                cached_data = {"samples": folder_samples, "buckets": folder_buckets}
                with open(cache_file, "wb") as f:
                    pickle.dump(cached_data, f)
            print(f"Cached {len(folder_samples)} samples from {folder}")
        else:
            print(f"Loading cached metadata from: {folder}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            folder_samples = cached_data["samples"]
            folder_buckets = cached_data["buckets"]
            print(f"Loaded {len(folder_samples)} samples from cache: {folder}")

        sample_idx_offset = len(self.samples)
        self.samples.extend(folder_samples)

        for bucket_key, indices in folder_buckets.items():
            adjusted_indices = [idx + sample_idx_offset for idx in indices]
            self.buckets[bucket_key].extend(adjusted_indices)

    def _build_folder_metadata(self, folder):
        feature_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
        samples = []
        buckets = defaultdict(list)
        sample_idx = 0

        print(f"Processing {len(feature_files)} files in {folder}...")

        for i, feature_file in enumerate(feature_files):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(feature_files)} files")

            feature_path = os.path.join(folder, feature_file)

            # Parse filename
            parts = feature_file.split("_")
            uttid = "_".join(parts[:-3])
            num_frame = int(parts[-3])
            height = int(parts[-2])
            width = int(parts[-1].replace(".pt", ""))

            # keep length >= 121
            if num_frame < 121:
                continue

            # keep resolution
            allowed_resolutions = [
                (self.single_height, self.single_width),
                (self.single_height // 2, self.single_width // 2),
                (self.single_height // 4, self.single_width // 4),
            ]
            if self.single_res and (height, width) not in allowed_resolutions:
                continue

            bucket_key = (num_frame, height, width)

            sample_info = {
                "uttid": uttid,
                "dataset_name": folder.rstrip("/"),
                "file_path": feature_path,
                "bucket_key": bucket_key,
                "num_frame": num_frame,
                "height": height,
                "width": width,
            }

            samples.append(sample_info)
            buckets[bucket_key].append(sample_idx)
            sample_idx += 1

        return samples, buckets

    def set_epoch(self, epoch):
        self._epoch = epoch

    def prepare_stage1_latent(self, vae_latent, idx, base_vae_latent=None):
        source_latent = base_vae_latent if base_vae_latent is not None else vae_latent

        x0_latent = None
        if self.is_keep_x0:
            x0_latent = source_latent[0, :, :1, :, :].clone()
        total_sections = source_latent.shape[0]
        latent_window_size = source_latent.shape[2]
        history_window_size = sum(self.history_sizes)
        section_size = history_window_size + latent_window_size

        temp_source_latent = rearrange(source_latent, "b c t h w -> c (b t) h w")
        zero_padding_source = torch.zeros(
            temp_source_latent.shape[0],
            history_window_size,
            temp_source_latent.shape[2],
            temp_source_latent.shape[3],
            device=temp_source_latent.device,
            dtype=temp_source_latent.dtype,
        )
        continue_source_latent = torch.cat([zero_padding_source, temp_source_latent], dim=1)

        temp_vae_latent = rearrange(vae_latent, "b c t h w -> c (b t) h w")
        zero_padding_vae = torch.zeros(
            temp_vae_latent.shape[0],
            history_window_size,
            temp_vae_latent.shape[2],
            temp_vae_latent.shape[3],
            device=temp_vae_latent.device,
            dtype=temp_vae_latent.dtype,
        )
        continue_vae_latent = torch.cat([zero_padding_vae, temp_vae_latent], dim=1)

        sample_seed = self.base_seed + self._epoch * 1000000 + idx
        choice_idx = torch.randint(
            1, total_sections, (1,), generator=torch.Generator().manual_seed(sample_seed)
        ).item()
        if choice_idx == 0 and x0_latent is not None:
            x0_latent = torch.zeros_like(x0_latent)

        clean_all_vae_latent = None
        if self.return_all_vae_latent:
            max_start_idx = total_sections - self.num_rollout_sections
            if max_start_idx < 0:
                raise ValueError(
                    f"Not enough sections: total_sections={total_sections}, num_rollout_sections={self.num_rollout_sections}"
                )
            start_section_idx = random.randint(0, max_start_idx)
            start_indice = start_section_idx * latent_window_size
            end_indice = start_indice + history_window_size + self.num_rollout_sections * latent_window_size
            clean_all_vae_latent = continue_source_latent[:, start_indice:end_indice, :, :]

        start_indice = choice_idx * latent_window_size
        end_indice = start_indice + section_size

        history_latent = continue_source_latent[:, start_indice : start_indice + history_window_size, :, :]
        target_latent = continue_vae_latent[:, start_indice + history_window_size : end_indice, :, :]

        return x0_latent, history_latent, target_latent, clean_all_vae_latent

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_f = self.samples[idx]["num_frame"]
        anchor_h = self.samples[idx]["height"]
        anchor_w = self.samples[idx]["width"]
        while True:
            sample_info = self.samples[idx]
            if (
                anchor_f != sample_info["num_frame"]
                or anchor_h != sample_info["height"]
                or anchor_w != sample_info["width"]
            ):
                idx = random.randint(0, len(self.samples) - 1)
                print("Try to find a same dim sample, retrying...")
                continue
            try:
                base_vae_latent = None
                if (anchor_h, anchor_w) in [
                    (self.single_height // 2, self.single_width // 2),
                    (self.single_height // 4, self.single_width // 4),
                ]:
                    base_file_path = (
                        sample_info["file_path"]
                        .replace("/mid", "")
                        .replace("/low", "")
                        .replace(
                            f"{self.single_height // 2}_{self.single_width // 2}",
                            f"{self.single_height}_{self.single_width}",
                        )
                        .replace(
                            f"{self.single_height // 4}_{self.single_width // 4}",
                            f"{self.single_height}_{self.single_width}",
                        )
                    )
                    base_vae_latent = torch.load(base_file_path, map_location="cpu", weights_only=False)["vae_latent"]

                feature_data = torch.load(sample_info["file_path"], map_location="cpu", weights_only=False)
                x0_latent, history_latent, target_latent, clean_all_vae_latent = self.prepare_stage1_latent(
                    feature_data["vae_latent"], idx, base_vae_latent
                )
                if self.return_prompt_raw:
                    prompt_raws = feature_data["prompt_raw"]
                break
            except Exception:
                idx = random.randint(0, len(self.samples) - 1)
                print(f"Error loading {sample_info['file_path']}, retrying...")
                file_name = os.path.basename(sample_info["file_path"])
                txt_name = f"{file_name}.txt"
                with open(txt_name, "w") as f:
                    f.write(sample_info["file_path"] + "\n")

        output_dict = {
            "uttid": sample_info["uttid"],
            "bucket_key": sample_info["bucket_key"],
            "dataset_name": sample_info["dataset_name"],
            "num_frame": sample_info["num_frame"],
            "height": sample_info["height"],
            "width": sample_info["width"],
            "x0_latents": x0_latent,
            "history_latents": history_latent,
            "target_latents": target_latent,
            "clean_all_latents": clean_all_vae_latent,
            "prompt_embeds": feature_data["prompt_embed"],
            "prompt_attention_masks": feature_data.get("prompt_attention_mask", None),
        }

        if self.return_prompt_raw:
            output_dict["prompt_raws"] = prompt_raws

        return output_dict


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

    feature_folder = [
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v1_2/latents-fp9-384_0.01-0.015_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v2/latents-fp9-384_0.01-0.015_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v4/latents-fp9-384_0.01-0.015_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v1_2/latents-fp9-384_0.015-0.02_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v2/latents-fp9-384_0.015-0.02_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v4/latents-fp9-384_0.015-0.02_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v1_2/latents-fp9-384_0.02_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v2/latents-fp9-384_0.02_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v4/latents-fp9-384_0.02_with_prompt",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-pexels-45k/latents-fp9-384_with_prompt",
    ]
    dataloader_num_workers = 96
    batch_size = 8
    num_train_epochs = 1
    seed = 0
    output_dir = "accelerate_checkpoints"
    checkpoint_dirs = (
        [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if os.path.exists(output_dir)
        else []
    )

    dataset_ratios = {}
    # dataset_ratios = {
    #     "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v4/latents": 0.9,
    #     "/mnt/hdfs/data/ysh_new/userful_things_wan/sekai/sekai-real-walking-hq-193/latents_stride1": 0.1
    # }

    # maybe_init_distributed_environment_and_model_parallel(1,1)
    accelerator = Accelerator()

    # print(get_world_rank(), get_world_size(), get_sp_world_size())
    print(accelerator.process_index, accelerator.num_processes)

    dataset = BucketedFeatureDataset(
        feature_folder,
        force_rebuild=True,
        return_all_vae_latent=False,
        return_prompt_raw=False,
        single_res=True,
        single_height=384,
        single_width=640,
        seed=seed,
    )
    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        dataset_sampling_ratios=dataset_ratios,
        seed=seed,
        # num_sp_groups=get_world_size() // get_sp_world_size(),
        # sp_world_size=get_sp_world_size(),
        # global_rank=get_world_rank(),
        num_sp_groups=accelerator.num_processes // 1,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )
    # dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=dataloader_num_workers)
    dataloader = StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    print(len(dataset), len(dataloader))
    # dataloader = accelerator.prepare(dataloader)
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")
    # print(f"Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")

    max_samples = 500000
    output_txt = "latent_statistics.txt"

    # 初始化统计变量
    stats = {
        "x0_latents": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "history_latents": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "target_latents": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
    }

    sampled_count = 0

    print(f"Starting to collect statistics from {max_samples} randomly sampled videos...")

    if accelerator.is_main_process:
        pbar = tqdm(total=max_samples, desc="Collecting statistics", unit="videos")

    for epoch in range(num_train_epochs):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)

        for i, batch in enumerate(dataloader):
            if accelerator.is_main_process:
                # 获取latents（已经是shuffle后的随机样本）
                x0_latents = batch["x0_latents"].cpu().float()
                history_latents = batch["history_latents"].cpu().float()
                target_latents = batch["target_latents"].cpu().float()

                # 统计 x0_latents
                stats["x0_latents"]["sum"] += x0_latents.sum().item()
                stats["x0_latents"]["sum_sq"] += (x0_latents**2).sum().item()
                stats["x0_latents"]["count"] += x0_latents.numel()

                # 统计 history_latents
                stats["history_latents"]["sum"] += history_latents.sum().item()
                stats["history_latents"]["sum_sq"] += (history_latents**2).sum().item()
                stats["history_latents"]["count"] += history_latents.numel()

                # 统计 target_latents
                stats["target_latents"]["sum"] += target_latents.sum().item()
                stats["target_latents"]["sum_sq"] += (target_latents**2).sum().item()
                stats["target_latents"]["count"] += target_latents.numel()

                sampled_count += x0_latents.shape[0]

                pbar.update(batch_size)

                if sampled_count % 1000 == 0:
                    print(f"Sampled {sampled_count}/{max_samples} videos...")

                # 达到1万条就停止
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
        output_lines.append(f"Dataset: {feature_folder[0]}")
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
