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

import json
import os
import pickle
import random
from collections import defaultdict
from typing import Optional

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, Sampler

from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video


class PyVideoReader:
    def __init__(self, path, threads=0):
        self.path = path
        self.threads = threads
        self.frames = iio.imread(path, plugin="FFMPEG")

        if self.frames.ndim == 3:
            self.frames = self.frames[..., None]

        self.length = self.frames.shape[0]

    def __len__(self):
        return self.length

    def get_batch(self, indices):
        indices = list(indices)
        return np.asarray(self.frames[indices])

    def __getitem__(self, idx):
        return self.frames[idx]


resolution_bucket_options = {
    640: [
        (768, 320),
        (768, 384),
        (640, 384),
        (768, 512),
        (576, 448),
        (512, 512),
        (448, 576),
        (512, 768),
        (384, 640),
        (384, 768),
        (320, 768),
    ],
    "giga_ctrl": [
        (1920, 480),
        (480, 1920),
    ],
}


length_bucket_options = {
    1: [
        501,
        481,
        461,
        441,
        421,
        401,
        381,
        361,
        341,
        321,
        301,
        281,
        261,
        241,
        221,
        193,
        181,
        161,
        141,
        121,
        101,
        81,
        61,
        41,
        21,
    ],
    2: [193, 177, 161, 156, 145, 133, 129, 121, 113, 109, 97, 85, 81, 73, 65, 61, 49, 37, 25],
}


def find_nearest_resolution_bucket(h, w, resolution=640):
    min_metric = float("inf")
    best_bucket = None

    for bucket_h, bucket_w in resolution_bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)

    return best_bucket


def find_nearest_length_bucket(length, stride=1):
    buckets = length_bucket_options[stride]
    min_bucket = min(buckets)

    if length < min_bucket:
        return length

    valid_buckets = [bucket for bucket in buckets if bucket <= length]
    return max(valid_buckets)


def _read_one_video_cut_crop_resize(
    video_path,
    f_prime,
    h_prime,
    w_prime,
    stride=1,
    start_frame=None,
    end_frame=None,
    crop=None,
):
    frame_indices = list(range(start_frame, end_frame, stride))
    assert len(frame_indices) == f_prime, f"{len(frame_indices)} != {f_prime}"

    vr = PyVideoReader(video_path, threads=0)

    if max(frame_indices) >= len(vr):
        raise IndexError(
            f"video too short: path={video_path}, len={len(vr)}, max_idx={max(frame_indices)}"
        )

    frames = torch.from_numpy(vr.get_batch(frame_indices)).float()

    frames = (frames / 127.5) - 1.0
    video = frames.permute(0, 3, 1, 2)

    s_x, e_x, s_y, e_y = crop
    video = video[:, :, s_y:e_y, s_x:e_x]

    frames, channels, h, w = video.shape
    aspect_ratio_original = h / w
    aspect_ratio_target = h_prime / w_prime

    if aspect_ratio_original >= aspect_ratio_target:
        new_h = int(w * aspect_ratio_target)
        top = (h - new_h) // 2
        bottom = top + new_h
        left = 0
        right = w
    else:
        new_w = int(h / aspect_ratio_target)
        left = (w - new_w) // 2
        right = left + new_w
        top = 0
        bottom = h

    cropped_video = video[:, :, top:bottom, left:right]
    resized_video = torchvision.transforms.functional.resize(
        cropped_video,
        (h_prime, w_prime),
    )

    return resized_video


def read_cut_crop_and_resize(
    video_path,
    control_video_path,
    f_prime,
    h_prime,
    w_prime,
    stride=1,
    start_frame=None,
    end_frame=None,
    crop=None,
):
    video_data = _read_one_video_cut_crop_resize(
        video_path=video_path,
        f_prime=f_prime,
        h_prime=h_prime,
        w_prime=w_prime,
        stride=stride,
        start_frame=start_frame,
        end_frame=end_frame,
        crop=crop,
    )

    control_video_data = _read_one_video_cut_crop_resize(
        video_path=control_video_path,
        f_prime=f_prime,
        h_prime=h_prime,
        w_prime=w_prime,
        stride=stride,
        start_frame=start_frame,
        end_frame=end_frame,
        crop=crop,
    )

    return video_data, control_video_data


def save_frames(frame_raw, fps=24, video_path="1.mp4"):
    save_list = []

    for frame in frame_raw:
        frame = (frame + 1) / 2 * 255
        frame = torchvision.transforms.transforms.ToPILImage()(frame.to(torch.uint8)).convert("RGB")
        save_list.append(frame)

    export_to_video(save_list, video_path, fps=fps)

    del save_list
    free_memory()


class BucketedFeatureDataset(Dataset):
    def __init__(
        self,
        json_files,
        video_folders,
        control_video_folders=None,
        stride=1,
        base_fps=None,
        resolution=640,
        force_rebuild=True,
        single_res=False,
        single_length=False,
        single_num_frame=81,
        single_height=384,
        single_width=640,
        multi_res=False,
        id_token: Optional[str] = None,
    ):
        self.stride = stride
        self.base_fps = base_fps
        self.resolution = resolution
        self.force_rebuild = force_rebuild
        self.single_res = single_res
        self.single_height = single_height
        self.single_width = single_width
        self.single_length = single_length
        self.single_num_frame = single_num_frame
        self.multi_res = multi_res
        self.id_token = id_token or ""
        self._epoch = 0

        self.json_files = [json_files] if isinstance(json_files, str) else json_files
        self.video_folders = [video_folders] if isinstance(video_folders, str) else video_folders

        if control_video_folders is None:
            self.control_video_folders = self.video_folders
        else:
            self.control_video_folders = (
                [control_video_folders]
                if isinstance(control_video_folders, str)
                else control_video_folders
            )

        assert len(self.json_files) == len(self.video_folders)
        assert len(self.json_files) == len(self.control_video_folders)

        self.samples = []
        self.buckets = defaultdict(list)

        for json_file, video_folder, control_video_folder in zip(
            self.json_files,
            self.video_folders,
            self.control_video_folders,
        ):
            cache_file = json_file.replace(".json", "_cache.pkl").replace(".csv", "_cache.pkl")
            self._process_json_file(
                json_file=json_file,
                video_folder=video_folder,
                control_video_folder=control_video_folder,
                cache_file=cache_file,
            )

    def _process_json_file(self, json_file, video_folder, control_video_folder, cache_file):
        if self.force_rebuild or not os.path.exists(cache_file):
            if os.path.exists(cache_file):
                print(f"Remove {cache_file}")
                os.remove(cache_file)

            print(f"Building metadata cache for file: {json_file}")
            print(f"  Video folder: {video_folder}")
            print(f"  Control video folder: {control_video_folder}")

            file_samples, file_buckets = self._build_file_metadata(
                json_file=json_file,
                video_folder=video_folder,
                control_video_folder=control_video_folder,
            )

            print(f"Saving metadata cache to: {cache_file}")
            cached_data = {
                "samples": file_samples,
                "buckets": file_buckets,
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)

            print(f"Cached {len(file_samples)} samples from {json_file}\n")

        else:
            print(f"Loading cached metadata from: {cache_file}")

            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)

            file_samples = cached_data["samples"]
            file_buckets = cached_data["buckets"]

            print(f"Loaded {len(file_samples)} samples from cache: {cache_file}\n")

        sample_idx_offset = len(self.samples)
        self.samples.extend(file_samples)

        for bucket_key, indices in file_buckets.items():
            adjusted_indices = [idx + sample_idx_offset for idx in indices]
            self.buckets[bucket_key].extend(adjusted_indices)

    def _normalize_video_rel_path(self, path):
        return (
            path.replace("videos_clip_v1_20241111/", "")
            .replace("videos_clip_v2_20241111/", "")
            .replace("videos_clip_v4_20241111/", "")
        )

    def _build_file_metadata(self, json_file, video_folder, control_video_folder):
        # ====================== 🔥 自动兼容 JSON / JSONL ======================
        data = []
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                tmp_data = json.load(f)
                if isinstance(tmp_data, list):
                    data = tmp_data
                    print(f"✅ 加载标准 JSON 数组: {json_file}")
                else:
                    data = [tmp_data]
                    print(f"✅ 加载标准 JSON 对象: {json_file}")
        except json.JSONDecodeError:
            print(f"ℹ️  非标准JSON，尝试按 JSONL 加载: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        # 单独捕获单行错误，只跳过坏行，不整体崩溃
                        print(f"⚠️  跳过非法行 {line_idx}: {str(e)[:150]}")
                        continue
        except Exception as e:
            # 其他致命错误（文件不存在/权限等）
            print(f"❌ 读取文件失败 {json_file}: {e}")
            return []
        # ====================================================================

        print(f"Scanning video folder: {video_folder}")
        existing_videos = set()

        for root, dirs, files in os.walk(video_folder):
            for file in files:
                if file.endswith(".mp4"):
                    rel_path = os.path.relpath(os.path.join(root, file), video_folder)
                    existing_videos.add(rel_path)

        print(f"Found {len(existing_videos)} video files")

        print(f"Scanning control video folder: {control_video_folder}")
        existing_control_videos = set()

        for root, dirs, files in os.walk(control_video_folder):
            for file in files:
                if file.endswith(".mp4"):
                    rel_path = os.path.relpath(os.path.join(root, file), control_video_folder)
                    existing_control_videos.add(rel_path)

        print(f"Found {len(existing_control_videos)} control video files")

        rows = []

        for item in data:
            if "control_path" not in item:
                raise KeyError(
                    "json item missing `control_path`. "
                    "Please add control_path for each sample."
                )

            rows.append(
                {
                    "cut": item["cut"],
                    "crop": item["crop"],
                    "path": item["path"],
                    "control_path": item["control_path"],
                    "num_frames": item["num_frames"],
                    "width": item["resolution"]["width"],
                    "height": item["resolution"]["height"],
                    "fps": item["fps"],
                    "cap": item["cap"],
                }
            )

        df = pd.DataFrame(rows)

        samples = []
        buckets = defaultdict(list)
        sample_idx = 0

        print(f"Processing {len(df)} records from {json_file} with stride={self.stride}...")

        for i, row in df.iterrows():
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(df)} records")

            video_file = self._normalize_video_rel_path(row["path"])

            if video_file not in existing_videos:
                print(f"bad video: {video_file}")
                continue

            video_path = os.path.join(video_folder, video_file)

            control_video_file = self._normalize_video_rel_path(row["control_path"])

            if control_video_file not in existing_control_videos:
                print(f"bad control video: {control_video_file}")
                continue

            control_video_path = os.path.join(control_video_folder, control_video_file)

            cut_start_frame = int(row["cut"][0])
            cut_end_frame = int(row["cut"][1])
            num_frame = cut_end_frame - cut_start_frame

            if self.single_length:
                if num_frame < self.single_num_frame:
                    continue
            else:
                if num_frame < 121:
                    continue

            uttid = os.path.basename(video_file).replace(".mp4", "") + f"_{cut_start_frame}-{cut_end_frame}"
            fps = row["fps"]

            crop = row["crop"]
            width = int(crop[1] - crop[0])
            height = int(crop[3] - crop[2])

            cap = row["cap"]
            if isinstance(cap, list):
                prompt = cap[0] if len(cap) > 0 else ""
            else:
                prompt = str(cap)

            effective_num_frame = (num_frame + self.stride - 1) // self.stride
            bucket_num_frame = find_nearest_length_bucket(
                effective_num_frame,
                stride=self.stride,
            )

            bucket_height, bucket_width = find_nearest_resolution_bucket(
                height,
                width,
                resolution=self.resolution,
            )

            if self.single_res or self.multi_res:
                allowed_resolutions = [(self.single_height, self.single_width)]

                if self.multi_res:
                    allowed_resolutions.extend(
                        [
                            (self.single_height // 2, self.single_width // 2),
                            (self.single_height // 4, self.single_width // 4),
                        ]
                    )

                if (bucket_height, bucket_width) not in allowed_resolutions:
                    print(
                        f"[❌ Error] continue res {bucket_height}x{bucket_width} "
                        f"not in allowed {allowed_resolutions}"
                    )
                    continue

                bucket_height, bucket_width = random.choice(allowed_resolutions)

            if self.single_length:
                bucket_num_frame = self.single_num_frame

            if self.base_fps is not None:
                stride = max(int(fps / self.base_fps), 1)
                required_frames = bucket_num_frame * stride

                if required_frames >= num_frame:
                    print("continue frame")
                    continue
            else:
                stride = self.stride

            bucket_key = (bucket_num_frame, bucket_height, bucket_width)

            sample_info = {
                "uttid": uttid,
                "dataset_name": json_file.rstrip("/"),
                "video_folder": video_folder,
                "control_video_folder": control_video_folder,
                "video_path": video_path,
                "control_video_path": control_video_path,
                "bucket_key": bucket_key,
                "prompt": self.id_token + prompt,
                "fps": fps,
                "stride": stride,
                "effective_num_frame": effective_num_frame,
                "num_frame": num_frame,
                "height": height,
                "width": width,
                "bucket_num_frame": bucket_num_frame,
                "bucket_height": bucket_height,
                "bucket_width": bucket_width,
                "cut_start_frame": cut_start_frame,
                "cut_end_frame": cut_end_frame,
                "crop": crop,
            }

            samples.append(sample_info)
            buckets[bucket_key].append(sample_idx)
            sample_idx += 1

        return samples, buckets

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_h = self.samples[idx]["bucket_height"]
        anchor_w = self.samples[idx]["bucket_width"]
        anchor_f = self.samples[idx]["bucket_num_frame"]

        max_retries = 1000
        retry_count = 0

        while retry_count < max_retries:
            sample_info = self.samples[idx]

            if (
                anchor_h != sample_info["bucket_height"]
                or anchor_w != sample_info["bucket_width"]
                or anchor_f != sample_info["bucket_num_frame"]
            ):
                idx = random.randint(0, len(self.samples) - 1)
                retry_count += 1
                continue

            try:
                stride = sample_info["stride"]
                cut_start_frame = sample_info["cut_start_frame"]
                cut_end_frame = sample_info["cut_end_frame"]
                bucket_num_frame = sample_info["bucket_num_frame"]

                max_start_frame = cut_end_frame - bucket_num_frame * stride

                if max_start_frame < cut_start_frame:
                    start_frame = cut_start_frame
                else:
                    start_frame = random.randint(cut_start_frame, max_start_frame)

                end_frame = start_frame + bucket_num_frame * stride

                video_data, control_video_data = read_cut_crop_and_resize(
                    video_path=sample_info["video_path"],
                    control_video_path=sample_info["control_video_path"],
                    f_prime=sample_info["bucket_num_frame"],
                    h_prime=sample_info["bucket_height"],
                    w_prime=sample_info["bucket_width"],
                    stride=stride,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    crop=sample_info["crop"],
                )

                return {
                    "uttid": sample_info["uttid"],
                    "bucket_key": sample_info["bucket_key"],
                    "dataset_name": sample_info["dataset_name"],
                    "video_metadata": {
                        "num_frames": sample_info["bucket_num_frame"],
                        "height": sample_info["bucket_height"],
                        "width": sample_info["bucket_width"],
                        "fps": sample_info["fps"],
                        "stride": stride,
                        "effective_num_frame": sample_info["effective_num_frame"],
                    },
                    "videos": video_data,
                    "control_videos": control_video_data,
                    "prompts": sample_info["prompt"],
                    "first_frames_images": (video_data[0] + 1) / 2 * 255,
                }

            except Exception as e:
                print(f"Error loading {sample_info['video_path']}: {repr(e)}")
                print(f"Error control {sample_info.get('control_video_path')}")
                idx = random.randint(0, len(self.samples) - 1)
                retry_count += 1

        print(f"Failed to load sample after {max_retries} retries, returning None")
        return None


class BucketedSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False,
        shuffle=True,
        seed=42,
        dataset_sampling_ratios=None,
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.generator = torch.Generator()
        self.buckets = dataset.buckets
        self._epoch = 0

        self.num_sp_groups = num_sp_groups
        self.sp_world_size = sp_world_size
        self.global_rank = global_rank
        self.ith_sp_group = self.global_rank // self.sp_world_size

        self.dataset_sampling_ratios = (
            {key.rstrip("/"): value for key, value in dataset_sampling_ratios.items()}
            if dataset_sampling_ratios is not None
            else {}
        )

        self._prepare_dataset_buckets()

    def _prepare_dataset_buckets(self):
        self.dataset_buckets = {}

        for bucket_key, sample_indices in self.buckets.items():
            dataset_groups = {}

            for idx in sample_indices:
                dataset_name = self.dataset.samples[idx]["dataset_name"]

                if dataset_name not in dataset_groups:
                    dataset_groups[dataset_name] = []

                dataset_groups[dataset_name].append(idx)

            self.dataset_buckets[bucket_key] = dataset_groups

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _shard_indices_for_sp_group(self, indices):
        if self.num_sp_groups == 1:
            return indices

        if isinstance(indices, list):
            indices_tensor = torch.tensor(indices, dtype=torch.long)
        else:
            indices_tensor = indices

        total_size = len(indices_tensor)

        if total_size % self.num_sp_groups != 0:
            if not self.drop_last:
                padding_size = self.num_sp_groups - (total_size % self.num_sp_groups)
                indices_tensor = torch.cat([indices_tensor, indices_tensor[:padding_size]])
        else:
            if self.drop_last:
                truncate_size = (total_size // self.num_sp_groups) * self.num_sp_groups
                indices_tensor = indices_tensor[:truncate_size]

        sp_group_indices = indices_tensor[self.ith_sp_group :: self.num_sp_groups]
        return sp_group_indices.tolist()

    def _apply_global_ratio_sampling(self):
        if not self.dataset_sampling_ratios:
            return

        dataset_sample_map = {}

        for bucket_key, dataset_groups in self.dataset_buckets.items():
            for dataset_name, indices in dataset_groups.items():
                if dataset_name not in dataset_sample_map:
                    dataset_sample_map[dataset_name] = {
                        "indices": [],
                        "buckets": [],
                    }

                dataset_sample_map[dataset_name]["indices"].extend(indices)
                dataset_sample_map[dataset_name]["buckets"].extend([bucket_key] * len(indices))

        total_samples = sum(len(info["indices"]) for info in dataset_sample_map.values())
        total_ratio = sum(self.dataset_sampling_ratios.values())

        sampled_dataset_map = {}

        for dataset_name, info in dataset_sample_map.items():
            if dataset_name in self.dataset_sampling_ratios:
                ratio = self.dataset_sampling_ratios[dataset_name] / total_ratio
                target_samples = max(1, int(total_samples * ratio))

                indices = info["indices"]
                buckets = info["buckets"]

                if len(indices) >= target_samples:
                    selected = torch.randperm(
                        len(indices),
                        generator=self.generator,
                    )[:target_samples].tolist()

                    sampled_indices = [indices[i] for i in selected]
                    sampled_buckets = [buckets[i] for i in selected]

                else:
                    sampled_indices = []
                    sampled_buckets = []
                    remaining = target_samples

                    while remaining > 0:
                        repeat_count = min(remaining, len(indices))

                        selected = torch.randperm(
                            len(indices),
                            generator=self.generator,
                        )[:repeat_count].tolist()

                        sampled_indices.extend([indices[i] for i in selected])
                        sampled_buckets.extend([buckets[i] for i in selected])
                        remaining -= repeat_count

                sampled_dataset_map[dataset_name] = {
                    "indices": sampled_indices,
                    "buckets": sampled_buckets,
                }

            else:
                sampled_dataset_map[dataset_name] = info

        new_dataset_buckets = {}

        for bucket_key in self.dataset_buckets.keys():
            new_dataset_buckets[bucket_key] = {}

        for dataset_name, info in sampled_dataset_map.items():
            indices = info["indices"]
            buckets = info["buckets"]

            for idx, bucket_key in zip(indices, buckets):
                if dataset_name not in new_dataset_buckets[bucket_key]:
                    new_dataset_buckets[bucket_key][dataset_name] = []

                new_dataset_buckets[bucket_key][dataset_name].append(idx)

        self.dataset_buckets = new_dataset_buckets

    def __iter__(self):
        epoch_seed = self.seed + self._epoch
        self.generator.manual_seed(epoch_seed)

        if self.dataset_sampling_ratios:
            self._apply_global_ratio_sampling()

        bucket_iterators = {}
        bucket_batches = {}

        for bucket_key, dataset_groups in self.dataset_buckets.items():
            balanced_indices = self._create_balanced_indices(dataset_groups)

            if self.shuffle:
                perm = torch.randperm(
                    len(balanced_indices),
                    generator=self.generator,
                ).tolist()

                balanced_indices = [balanced_indices[i] for i in perm]

            sp_group_indices = self._shard_indices_for_sp_group(balanced_indices)

            batches = []

            for i in range(0, len(sp_group_indices), self.batch_size):
                batch = sp_group_indices[i : i + self.batch_size]

                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

            if batches:
                bucket_batches[bucket_key] = batches
                bucket_iterators[bucket_key] = iter(batches)

        remaining_buckets = list(bucket_iterators.keys())

        while remaining_buckets:
            idx = torch.randint(
                len(remaining_buckets),
                (1,),
                generator=self.generator,
            ).item()

            bucket_key = remaining_buckets[idx]
            bucket_iter = bucket_iterators[bucket_key]

            try:
                batch = next(bucket_iter)
                yield batch
            except StopIteration:
                remaining_buckets.remove(bucket_key)

    def _create_balanced_indices(self, dataset_groups):
        return sum(dataset_groups.values(), [])

    def _equal_sampling(self, dataset_groups):
        all_indices = []
        dataset_names = list(dataset_groups.keys())

        if len(dataset_names) <= 1:
            return sum(dataset_groups.values(), [])

        min_samples = min(len(indices) for indices in dataset_groups.values())

        for dataset_name, indices in dataset_groups.items():
            if len(indices) > min_samples:
                selected = torch.randperm(
                    len(indices),
                    generator=self.generator,
                )[:min_samples].tolist()

                sampled_indices = [indices[i] for i in selected]
            else:
                sampled_indices = indices

            all_indices.extend(sampled_indices)

        return all_indices

    def _ratio_sampling(self, dataset_groups):
        return sum(dataset_groups.values(), [])

    def __len__(self):
        if self.dataset_sampling_ratios:
            temp_generator = torch.Generator()
            temp_generator.manual_seed(self.seed)

            dataset_sample_map = {}

            for bucket_key, dataset_groups in self.dataset_buckets.items():
                for dataset_name, indices in dataset_groups.items():
                    if dataset_name not in dataset_sample_map:
                        dataset_sample_map[dataset_name] = []

                    dataset_sample_map[dataset_name].extend(indices)

            total_samples = sum(len(indices) for indices in dataset_sample_map.values())
            total_ratio = sum(self.dataset_sampling_ratios.values())

            sampled_total = 0

            for dataset_name, indices in dataset_sample_map.items():
                if dataset_name in self.dataset_sampling_ratios:
                    ratio = self.dataset_sampling_ratios[dataset_name] / total_ratio
                    target_samples = max(1, int(total_samples * ratio))
                    sampled_total += target_samples
                else:
                    sampled_total += len(indices)

            sp_group_samples = sampled_total // self.num_sp_groups

            if not self.drop_last and sampled_total % self.num_sp_groups != 0:
                sp_group_samples += 1

            total_batches = sp_group_samples // self.batch_size

            if not self.drop_last and sp_group_samples % self.batch_size != 0:
                total_batches += 1

            return total_batches

        total_batches = 0

        for bucket_key, dataset_groups in self.dataset_buckets.items():
            balanced_indices = self._create_balanced_indices(dataset_groups)

            sp_group_size = len(balanced_indices) // self.num_sp_groups

            if not self.drop_last and len(balanced_indices) % self.num_sp_groups != 0:
                sp_group_size += 1

            num_batches = sp_group_size // self.batch_size

            if not self.drop_last and sp_group_size % self.batch_size != 0:
                num_batches += 1

            total_batches += num_batches

        return total_batches


def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    def collate_dict(data_list):
        if isinstance(data_list[0], dict):
            return {
                key: collate_dict([d[key] for d in data_list])
                for key in data_list[0]
            }
        elif isinstance(data_list[0], torch.Tensor):
            return torch.stack(data_list)
        else:
            return data_list

    return {
        key: collate_dict([d[key] for d in batch])
        for key in batch[0]
    }

if __name__ == "__main__":
    import os
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from collections import defaultdict

    # ============================================================
    # Config
    # ============================================================
    json_file = [
        "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view/helios_giga_ctrl.json",
    ]

    video_folder = [
        "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view/videos",
    ]

    control_video_folder = [
        "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view/control_videos",
    ]

    stride = 1
    batch_size = 2
    num_workers = 0
    seed = 42

    save_debug = True
    save_dir = "/mnt/pfs/users/zhanqian.wu/output/debug_bucketed_ctrl_dataset"
    os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # Dataset
    # ============================================================
    dataset = BucketedFeatureDataset(
        json_files=json_file,
        video_folders=video_folder,
        control_video_folders=control_video_folder,
        stride=stride,
        force_rebuild=True,
        resolution="giga_ctrl",
        single_res=True,
        single_height=480,
        single_width=1920,
        single_length=False,
        multi_res=False,
    )

    print("=" * 80)
    print(f"✅ Dataset size: {len(dataset)}")
    print(f"✅ Bucket num: {len(dataset.buckets)}")
    print("✅ Buckets:")
    for k, v in dataset.buckets.items():
        print(f"  {k}: {len(v)} samples")
    print("=" * 80)

    assert len(dataset) > 0, "❌ dataset is empty"

    # ============================================================
    # Test single sample
    # ============================================================
    print("\n🧪 Testing single sample...")

    sample = dataset[0]

    assert sample is not None, "❌ sample is None"
    assert "videos" in sample, "❌ missing videos"
    assert "control_videos" in sample, "❌ missing control_videos"
    assert "first_frames_images" in sample, "❌ missing first_frames_images"

    video = sample["videos"]
    control_video = sample["control_videos"]

    print(f"uttid: {sample['uttid']}")
    print(f"bucket_key: {sample['bucket_key']}")
    print(f"prompt: {sample['prompts']}")
    print(f"videos shape: {tuple(video.shape)}")
    print(f"control_videos shape: {tuple(control_video.shape)}")
    print(f"video range: {video.min().item():.4f}, {video.max().item():.4f}")
    print(f"control range: {control_video.min().item():.4f}, {control_video.max().item():.4f}")

    assert isinstance(video, torch.Tensor)
    assert isinstance(control_video, torch.Tensor)
    assert video.shape == control_video.shape, (
        f"❌ video/control shape mismatch: "
        f"{tuple(video.shape)} vs {tuple(control_video.shape)}"
    )
    assert video.ndim == 4, f"❌ videos should be [T,C,H,W], got {video.shape}"
    assert control_video.ndim == 4, f"❌ control_videos should be [T,C,H,W], got {control_video.shape}"
    assert video.shape[1] == 3, f"❌ video channel should be 3, got {video.shape[1]}"
    assert control_video.shape[1] == 3, f"❌ control channel should be 3, got {control_video.shape[1]}"

    print("✅ Single sample passed")

    # ============================================================
    # Sampler + DataLoader
    # ============================================================
    print("\n🧪 Testing dataloader...")

    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"✅ Dataloader batches: {len(dataloader)}")

    dataset_counts = defaultdict(int)

    for step, batch in enumerate(dataloader):
        assert batch is not None, "❌ batch is None"

        assert "videos" in batch, "❌ batch missing videos"
        assert "control_videos" in batch, "❌ batch missing control_videos"
        assert "prompts" in batch, "❌ batch missing prompts"
        assert "first_frames_images" in batch, "❌ batch missing first_frames_images"

        videos = batch["videos"]
        control_videos = batch["control_videos"]

        print("\n" + "-" * 80)
        print(f"Step {step}")
        print(f"uttid: {batch['uttid']}")
        print(f"bucket_key: {batch['bucket_key']}")
        print(f"videos: {tuple(videos.shape)}")
        print(f"control_videos: {tuple(control_videos.shape)}")
        print(f"first_frames_images: {tuple(batch['first_frames_images'].shape)}")
        print(f"prompt[0]: {batch['prompts'][0]}")

        assert videos.shape == control_videos.shape, (
            f"❌ batch video/control mismatch: "
            f"{tuple(videos.shape)} vs {tuple(control_videos.shape)}"
        )

        assert videos.ndim == 5, f"❌ videos should be [B,T,C,H,W], got {videos.shape}"
        assert control_videos.ndim == 5, f"❌ control_videos should be [B,T,C,H,W], got {control_videos.shape}"

        b, t, c, h, w = videos.shape
        assert c == 3, f"❌ video C should be 3, got {c}"
        assert control_videos.shape[2] == 3, f"❌ control C should be 3, got {control_videos.shape[2]}"

        num_frames = batch["video_metadata"]["num_frames"]
        heights = batch["video_metadata"]["height"]
        widths = batch["video_metadata"]["width"]

        assert all(nf == num_frames[0] for nf in num_frames), "❌ frame numbers not consistent"
        assert all(hh == heights[0] for hh in heights), "❌ heights not consistent"
        assert all(ww == widths[0] for ww in widths), "❌ widths not consistent"

        for dataset_name in batch["dataset_name"]:
            dataset_counts[dataset_name] += 1

        # ========================================================
        # Save debug video
        # ========================================================
        if save_debug and step < 3:
            rgb = videos[0]
            ctrl = control_videos[0]

            # [T,C,H,W] -> concat width
            vis = torch.cat([rgb, ctrl], dim=-1)

            save_path = os.path.join(save_dir, f"debug_batch{step:03d}_rgb_control.mp4")
            save_frames(vis, fps=16, video_path=save_path)
            print(f"🎬 saved debug video: {save_path}")

        if step >= 5:
            break

    print("\n" + "=" * 80)
    print("✅ All tests passed")
    print("📊 dataset counts:", dict(dataset_counts))
    print("=" * 80)