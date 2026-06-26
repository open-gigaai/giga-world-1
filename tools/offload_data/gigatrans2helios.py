import os
import json
import pickle
import hashlib
import random
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ============================================================
# Config
# ============================================================

DATA_ROOTS = [
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/task1",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/task2",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/task3",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/task4",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/task5",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/20250918T014_fold_shirt_task0116_cyt001_01",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/task11",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/merged",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/put_the_tableware_on_the_table",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/catch_object",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/clean_sink",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/microwave",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/stack_box",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/install_belt",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/stack_box_right_arm",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/chewing_gum_right_arm",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/fold_the_shirt_easy",

    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/fold_the_shirt_easy",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/fold_the_shirt_hard_fixed",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/fold_the_shirt_pants",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/fold_the_shirt_take4", #不太好用 部分有问题
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/heat_up_food",
    "/shared_disk/users/zhanqian.wu/data/train_data/giga_world_1_data/giga_aloha/ctrl/make_a_drink",
]

#SAVE_ROOT = "/mnt/pfs/users/zhanqian.wu/output/dataset"
SAVE_ROOT = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_trans_3view"

RGB_VIDEO_ROOT = os.path.join(SAVE_ROOT, "videos")
SOBEL_VIDEO_ROOT = os.path.join(SAVE_ROOT, "sobel_edge_videos")
DA2_DEPTH_VIDEO_ROOT = os.path.join(SAVE_ROOT, "da2_depth_videos")

SAVE_JSONL_PATH = os.path.join(SAVE_ROOT, "helios_giga_ctrl.jsonl")
SAVE_JSON_PATH = os.path.join(SAVE_ROOT, "helios_giga_ctrl.json")

NUM_FRAMES = int(os.environ.get("NUM_FRAMES", 121))
STRIDE = int(os.environ.get("STRIDE", 90))
FPS = int(os.environ.get("FPS", 10))  # NOTE: 24.0 是原始视频的 FPS

MAX_EPISODES = None

# 增量追加模式
RESET_JSONL = os.environ.get("RESET_JSONL", "0") == "1"
RESET_VIDEOS = os.environ.get("RESET_VIDEOS", "0") == "1"
ENABLE_APPEND_MODE = os.environ.get("ENABLE_APPEND_MODE", "1") == "1"

NUM_WORKERS_PER_GPU = int(os.environ.get("NUM_WORKERS_PER_GPU", 8))

# 手动指定 GPU 设备，例如 "0,1,2" 或 [0, 1, 2]
# 设为 None 则自动使用所有可用 GPU
# 也可以通过环境变量 CUDA_VISIBLE_DEVICES 覆盖
# GPU_IDS = None  # e.g. "0,1,2" or [0, 1, 2] or None
GPU_IDS = "0,3,4,5,6"

_env_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _env_cuda.strip():
    # 环境变量已设置，优先使用
    GPU_COUNT = len([x.strip() for x in _env_cuda.split(",") if x.strip()])
elif GPU_IDS is not None:
    if isinstance(GPU_IDS, str):
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in GPU_IDS)
    GPU_COUNT = len(GPU_IDS) if isinstance(GPU_IDS, list) else len(GPU_IDS.split(","))
else:
    GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

NUM_WORKERS = max(1, GPU_COUNT * NUM_WORKERS_PER_GPU)
DA2_BATCH_SIZE = int(os.environ.get("DA2_BATCH_SIZE", 4))

DA2_MODEL_SIZE = os.environ.get("DA2_MODEL_SIZE", "large")
HUGGINGFACE_MODEL_CACHE = "/shared_disk/models/huggingface"
DA2_MODEL_PATHS = {
    "small": os.path.join(HUGGINGFACE_MODEL_CACHE, "models--Depth-Anything-V2-Small-hf"),
    "base": os.path.join(HUGGINGFACE_MODEL_CACHE, "models--Depth-Anything-V2-Base-hf"),
    "large": os.path.join(HUGGINGFACE_MODEL_CACHE, "models--Depth-Anything-V2-Large-hf"),
}
DA2_MODELS = {}
DA2_MODEL_LOCK = threading.Lock()

VIEW_H = 480
VIEW_W = 640

OUT_H = 480
OUT_W = 640 * 3


# ============================================================
# Basic utils
# ============================================================

def mkdir(path):
    os.makedirs(path, exist_ok=True)


def parse_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def append_jsonl(path, item):
    with open(path, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def jsonl_to_json(jsonl_path, json_path):
    data = []

    if not os.path.exists(jsonl_path):
        return 0

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return len(data)


def build_unique_key_from_values(path, source_cut, prompt_type):
    if source_cut is None:
        source_cut = [0, 0]

    return (
        f"{path}|"
        f"{int(source_cut[0])}|"
        f"{int(source_cut[1])}|"
        f"{prompt_type}"
    )


def build_unique_key(item):
    return build_unique_key_from_values(
        path=item.get("path", ""),
        source_cut=item.get("source_cut", item.get("cut", [0, 0])),
        prompt_type=item.get("prompt_type", ""),
    )


def load_existing_keys(jsonl_path):
    keys = set()

    if not os.path.exists(jsonl_path):
        return keys

    bad_lines = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                keys.add(build_unique_key(item))
            except Exception:
                bad_lines += 1

    if bad_lines > 0:
        print(f"⚠️ load_existing_keys skipped bad lines: {bad_lines}")

    return keys


def resolve_path(root: str, path: str):
    if path is None:
        return None

    if os.path.isabs(path):
        return path

    return os.path.join(os.path.dirname(root), path)


def stable_episode_id(*paths):
    raw = "|".join([str(x) for x in paths if x])
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

    base = "episode"
    for p in paths:
        if p:
            base = os.path.splitext(os.path.basename(p))[0]
            break

    return f"{base}_{h}"


# ============================================================
# Prompt utils
# ============================================================

def choose_prompt_by_overlap(prompt_dict, clip_start, clip_end, key):
    """
    clip_end: exclusive
    prompt segment end_idx: inclusive
    """
    if prompt_dict is None:
        return ""

    if isinstance(prompt_dict, str):
        return prompt_dict

    if not isinstance(prompt_dict, dict) or len(prompt_dict) == 0:
        return ""

    best_text = ""
    best_overlap = -1
    best_center_dist = float("inf")

    clip_center = (clip_start + clip_end - 1) / 2.0

    for _, seg in prompt_dict.items():
        if not isinstance(seg, dict):
            continue

        seg_start = parse_int(seg.get("start_idx", 0))
        seg_end = parse_int(seg.get("end_idx", seg_start))
        text = seg.get(key, "")

        if not text:
            continue

        seg_end_exclusive = seg_end + 1

        overlap = max(
            0,
            min(clip_end, seg_end_exclusive) - max(clip_start, seg_start),
        )

        seg_center = (seg_start + seg_end) / 2.0
        center_dist = abs(clip_center - seg_center)

        if (
            overlap > best_overlap
            or (overlap == best_overlap and center_dist < best_center_dist)
        ):
            best_overlap = overlap
            best_center_dist = center_dist
            best_text = text

    return best_text


def choose_short_prompt(item, clip_start, clip_end):
    text = choose_prompt_by_overlap(
        item.get("short-prompt", {}),
        clip_start,
        clip_end,
        key="description",
    )

    if text:
        return text

    if item.get("prompt", ""):
        return item["prompt"]

    return ""


def choose_long_prompt(item, clip_start, clip_end):
    text = choose_prompt_by_overlap(
        item.get("long-prompt", {}),
        clip_start,
        clip_end,
        key="caption",
    )

    if text:
        return text

    return ""


def choose_background_prompt(item):
    text = item.get("background_prompt", "")
    if text:
        return text
    return ""


def join_prompt_with_background(prompt, background_prompt):
    prompt = str(prompt).strip()
    background_prompt = str(background_prompt).strip()

    if prompt and background_prompt:
        return f"{prompt} {background_prompt}"

    if prompt:
        return prompt

    if background_prompt:
        return background_prompt

    return ""


# ============================================================
# Video utils
# ============================================================

def load_video_clip_pil(video_path, start_idx, end_idx):
    """
    只读取当前 clip，不读取整段视频。
    """
    if video_path is None or not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")

    vr = VideoReader(video_path, ctx=cpu(0))

    video_len = len(vr)
    end_idx = min(end_idx, video_len)

    indices = list(range(start_idx, end_idx))
    frames_np = vr.get_batch(indices).asnumpy()

    return [Image.fromarray(x).convert("RGB") for x in frames_np]


def resize_keep_ratio_center_crop_pil(frames, target_h=480, target_w=640):
    out = []

    for img in frames:
        img = img.convert("RGB")
        w, h = img.size

        scale = max(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        img = img.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2

        img = img.crop((left, top, left + target_w, top + target_h))
        out.append(img)

    return out


def concat_three_views_horizontal(high, left, right):
    min_len = min(len(high), len(left), len(right))

    high = high[:min_len]
    left = left[:min_len]
    right = right[:min_len]

    out = []

    for h, l, r in zip(high, left, right):
        canvas = Image.new("RGB", (VIEW_W * 3, VIEW_H))
        canvas.paste(h, (0, 0))
        canvas.paste(l, (VIEW_W, 0))
        canvas.paste(r, (VIEW_W * 2, 0))
        out.append(canvas)

    return out


def save_pil_video(frames, save_path, fps=24.0):
    mkdir(os.path.dirname(save_path))

    frames_np = [np.asarray(x.convert("RGB")) for x in frames]

    imageio.mimsave(
        save_path,
        frames_np,
        fps=fps,
        codec="libx264",
        quality=8,
    )


def build_sobel_edge_clip(frames):
    edge_frames = []

    for img in frames:
        rgb = np.asarray(img.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(grad_x, grad_y)
        edge = np.clip(edge / (edge.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        edge = np.stack([edge] * 3, axis=-1)
        edge_frames.append(Image.fromarray(edge))

    return edge_frames


def get_task_device(task):
    if GPU_COUNT <= 0:
        return "cpu"
    worker_id = task.get("worker_id", 0)
    gpu_id = worker_id // NUM_WORKERS_PER_GPU
    return f"cuda:{gpu_id}"


def get_da2_model(device):
    if device in DA2_MODELS:
        return DA2_MODELS[device]

    with DA2_MODEL_LOCK:
        if device in DA2_MODELS:
            return DA2_MODELS[device]

        model_dir = DA2_MODEL_PATHS[DA2_MODEL_SIZE]
        image_processor = AutoImageProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(device, memory_format=torch.channels_last)
        model.eval()
        DA2_MODELS[device] = (model, image_processor)
        print(f"✅ Loaded DA2 model on {device}")
        return DA2_MODELS[device]


@torch.inference_mode()
def infer_da2_depth_batch(model, image_processor, images, device):
    images = [img.convert("RGB") for img in images]
    inputs = image_processor(
        images=images,
        return_tensors="pt",
        do_rescale=True,
    )
    inputs = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.autocast(device_type="cuda", dtype=model.dtype):
        outputs = model(**inputs)

    depth = outputs.predicted_depth.unsqueeze(1).contiguous(memory_format=torch.contiguous_format)
    target_h, target_w = images[0].height, images[0].width
    depth = F.interpolate(
        depth,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    d_min = depth.amin(dim=(1, 2), keepdim=True)
    d_max = depth.amax(dim=(1, 2), keepdim=True)
    depth = (depth - d_min) / (d_max - d_min + 1e-6)
    depth = (depth * 255).clamp(0, 255).to(torch.uint8)

    depth_np = depth.cpu().numpy()
    return [Image.fromarray(np.stack([x] * 3, axis=-1)) for x in depth_np]


def build_da2_depth_clip(frames, device):
    model, image_processor = get_da2_model(device)
    depth_frames = []

    for i in range(0, len(frames), DA2_BATCH_SIZE):
        batch = frames[i : i + DA2_BATCH_SIZE]
        depth_frames.extend(infer_da2_depth_batch(model, image_processor, batch, device))

    return depth_frames


def build_3view_rgb_clip(item, root, start_idx, end_idx):
    high_path = resolve_path(root, item.get("cam_high_video_path"))
    left_path = resolve_path(root, item.get("cam_left_wrist_video_path"))
    right_path = resolve_path(root, item.get("cam_right_wrist_video_path"))

    high = load_video_clip_pil(high_path, start_idx, end_idx)
    left = load_video_clip_pil(left_path, start_idx, end_idx)
    right = load_video_clip_pil(right_path, start_idx, end_idx)

    high = resize_keep_ratio_center_crop_pil(high, VIEW_H, VIEW_W)
    left = resize_keep_ratio_center_crop_pil(left, VIEW_H, VIEW_W)
    right = resize_keep_ratio_center_crop_pil(right, VIEW_H, VIEW_W)

    video = concat_three_views_horizontal(high, left, right)

    return video, high_path, left_path, right_path



# ============================================================
# Metadata utils
# ============================================================

def check_giga_ctrl_item(item):
    required_keys = [
        "cam_high_video_path",
        "cam_left_wrist_video_path",
        "cam_right_wrist_video_path",
    ]

    for k in required_keys:
        if not item.get(k, ""):
            return False

    return True



def build_clip_starts(video_length, num_frames, stride):
    return list(range(0, video_length - num_frames + 1, stride))


def build_helios_item(
    rel_rgb_path,
    rel_sobel_path,
    rel_da2_depth_path,
    caption,
    prompt_type,
    start_idx,
    end_idx,
    num_frames,
    episode_name,
    clip_id,
    source_info,
):
    return {
        "cut": [0, num_frames],
        "source_cut": [start_idx, end_idx],

        "crop": [0, OUT_W, 0, OUT_H],
        "fps": FPS,
        "num_frames": num_frames,

        "resolution": {
            "height": OUT_H,
            "width": OUT_W,
        },

        "cap": [caption],
        "prompt_type": prompt_type,

        "path": rel_rgb_path,
        "sobel_edge_path": rel_sobel_path,
        "da2_depth_path": rel_da2_depth_path,

        "episode_name": episode_name,
        "clip_id": clip_id,

        "view_mode": "3view_horizontal",
        "num_view": 3,

        "single_view_resolution": {
            "height": VIEW_H,
            "width": VIEW_W,
        },

        "source_info": source_info,
    }


# ============================================================
# Worker
# ============================================================

def process_single_clip(task):
    """
    return:
        {
            "ok": True/False,
            "meta_items": list[dict],
            "err": str
        }
    """
    root = task["root"]
    root_name = task["root_name"]
    item = task["item"]
    item_id = task["item_id"]
    clip_id = task["clip_id"]
    start_idx = task["start_idx"]
    end_idx = task["end_idx"]
    worker_id = task.get("worker_id", 0)
    device = get_task_device(task)

    try:
        if not check_giga_ctrl_item(item):
            return {
                "ok": False,
                "meta_items": [],
                "err": "missing required video/control keys",
            }

        episode_name = item.get(
            "episode_name",
            f"{root_name}_{item_id:06d}",
        )

        rgb_clip, high_path, left_path, right_path = build_3view_rgb_clip(
            item=item,
            root=root,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        real_len = len(rgb_clip)

        if real_len < NUM_FRAMES:
            return {
                "ok": False,
                "meta_items": [],
                "err": f"real_len too short: {real_len}",
            }

        rgb_clip = rgb_clip[:NUM_FRAMES]

        episode_id = stable_episode_id(
            high_path,
            left_path,
            right_path,
        )

        clip_name = f"{episode_id}_s{start_idx:06d}_e{end_idx:06d}.mp4"

        rgb_rel_path = os.path.join(root_name, clip_name)
        sobel_rel_path = os.path.join(root_name, clip_name)
        da2_depth_rel_path = os.path.join(root_name, clip_name)

        rgb_save_path = os.path.join(RGB_VIDEO_ROOT, rgb_rel_path)
        sobel_save_path = os.path.join(SOBEL_VIDEO_ROOT, sobel_rel_path)
        da2_depth_save_path = os.path.join(DA2_DEPTH_VIDEO_ROOT, da2_depth_rel_path)

        need_rgb = RESET_VIDEOS or not os.path.exists(rgb_save_path)
        need_sobel = RESET_VIDEOS or not os.path.exists(sobel_save_path)
        need_depth = RESET_VIDEOS or not os.path.exists(da2_depth_save_path)

        # 全部已存在则跳过推理
        if not need_rgb and not need_sobel and not need_depth:
            pass  # 直接进入 metadata 构建
        else:
            if need_rgb:
                save_pil_video(rgb_clip, rgb_save_path, fps=FPS)

            if need_sobel:
                sobel_clip = build_sobel_edge_clip(rgb_clip)
                save_pil_video(sobel_clip, sobel_save_path, fps=FPS)

            if need_depth:
                da2_depth_clip = build_da2_depth_clip(rgb_clip, device)
                save_pil_video(da2_depth_clip, da2_depth_save_path, fps=FPS)

        long_caption = choose_long_prompt(
            item=item,
            clip_start=start_idx,
            clip_end=end_idx,
        )

        short_caption = choose_short_prompt(
            item=item,
            clip_start=start_idx,
            clip_end=end_idx,
        )

        background_prompt = choose_background_prompt(item=item)

        short_caption = join_prompt_with_background(
            short_caption,
            background_prompt,
        )

        source_info = {
            "cam_high_video_path": os.path.abspath(high_path),
            "cam_left_wrist_video_path": os.path.abspath(left_path),
            "cam_right_wrist_video_path": os.path.abspath(right_path),
            "generated_sobel_edge": True,
            "generated_da2_depth": True,
            "da2_model_size": DA2_MODEL_SIZE,
        }

        meta_items = []

        if long_caption:
            meta_items.append(
                build_helios_item(
                    rel_rgb_path=rgb_rel_path,
                    rel_sobel_path=sobel_rel_path,
                    rel_da2_depth_path=da2_depth_rel_path,
                    caption=long_caption,
                    prompt_type="long",
                    start_idx=start_idx,
                    end_idx=end_idx,
                    num_frames=NUM_FRAMES,
                    episode_name=episode_name,
                    clip_id=clip_id,
                    source_info=source_info,
                )
            )

        if short_caption:
            meta_items.append(
                build_helios_item(
                    rel_rgb_path=rgb_rel_path,
                    rel_sobel_path=sobel_rel_path,
                    rel_da2_depth_path=da2_depth_rel_path,
                    caption=short_caption,
                    prompt_type="short",
                    start_idx=start_idx,
                    end_idx=end_idx,
                    num_frames=NUM_FRAMES,
                    episode_name=episode_name,
                    clip_id=clip_id,
                    source_info=source_info,
                )
            )

        return {
            "ok": True,
            "meta_items": meta_items,
            "err": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "meta_items": [],
            "err": repr(e),
        }


# ============================================================
# Main
# ============================================================

def build_all_tasks():
    tasks = []
    skipped_episodes = 0
    processed_episodes = 0
    skipped_clips_exist = 0

    for root in DATA_ROOTS:
        pkl_path = os.path.join(root, "labels", "data.pkl")
        new_pkl_path = os.path.join(root, "labels", "data_new.pkl")
        pkl_path = new_pkl_path if os.path.exists(new_pkl_path) else pkl_path

        if not os.path.exists(pkl_path):
            print(f"⚠️ skip missing: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)

        root_name = os.path.basename(root)

        for item_id, item in enumerate(tqdm(meta, desc=f"Scan {root_name}")):
            if MAX_EPISODES is not None and processed_episodes >= MAX_EPISODES:
                break

            if not check_giga_ctrl_item(item):
                skipped_episodes += 1
                continue

            video_length = item.get("video_length", None)
            if video_length is None:
                skipped_episodes += 1
                continue

            video_length = int(video_length)

            if video_length < NUM_FRAMES:
                skipped_episodes += 1
                continue

            clip_starts = build_clip_starts(
                video_length=video_length,
                num_frames=NUM_FRAMES,
                stride=STRIDE,
            )

            if len(clip_starts) == 0:
                skipped_episodes += 1
                continue

            # 预计算路径用于检查文件是否存在
            high_path = resolve_path(root, item.get("cam_high_video_path"))
            left_path = resolve_path(root, item.get("cam_left_wrist_video_path"))
            right_path = resolve_path(root, item.get("cam_right_wrist_video_path"))
            episode_id = stable_episode_id(high_path, left_path, right_path)

            for clip_id, start_idx in enumerate(clip_starts):
                end_idx = start_idx + NUM_FRAMES

                # 检查三个视频是否都已存在
                if not RESET_VIDEOS:
                    clip_name = f"{episode_id}_s{start_idx:06d}_e{end_idx:06d}.mp4"
                    rgb_ok = os.path.exists(os.path.join(RGB_VIDEO_ROOT, root_name, clip_name))
                    sobel_ok = os.path.exists(os.path.join(SOBEL_VIDEO_ROOT, root_name, clip_name))
                    depth_ok = os.path.exists(os.path.join(DA2_DEPTH_VIDEO_ROOT, root_name, clip_name))

                    if rgb_ok and sobel_ok and depth_ok:
                        skipped_clips_exist += 1
                        continue

                tasks.append(
                    {
                        "root": root,
                        "root_name": root_name,
                        "item": item,
                        "item_id": item_id,
                        "clip_id": clip_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                    }
                )

            processed_episodes += 1

    return tasks, processed_episodes, skipped_episodes, skipped_clips_exist


def main():
    mkdir(SAVE_ROOT)
    mkdir(RGB_VIDEO_ROOT)
    mkdir(SOBEL_VIDEO_ROOT)
    mkdir(DA2_DEPTH_VIDEO_ROOT)

    if RESET_JSONL and os.path.exists(SAVE_JSONL_PATH):
        os.remove(SAVE_JSONL_PATH)

    if ENABLE_APPEND_MODE and not RESET_JSONL:
        existing_keys = load_existing_keys(SAVE_JSONL_PATH)
    else:
        existing_keys = set()

    print(f"📦 existing json items: {len(existing_keys)}")

    tasks, processed_episodes, skipped_episodes_scan, skipped_clips_exist = build_all_tasks()
    random.shuffle(tasks)

    print("")
    print(f"🚀 total tasks/clips: {len(tasks)}")
    print(f"⏭️  skipped (exist):   {skipped_clips_exist}")
    print(f"🚀 gpu_count:         {GPU_COUNT}")
    print(f"🚀 cuda_devices:      {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print(f"🚀 workers_per_gpu:   {NUM_WORKERS_PER_GPU}")
    print(f"🚀 num_workers:       {NUM_WORKERS}")
    print(f"🚀 da2_model_size:    {DA2_MODEL_SIZE}")
    print(f"🚀 da2_batch_size:    {DA2_BATCH_SIZE}")
    print(f"🚀 output size:       {OUT_W}x{OUT_H}")
    print("")

    for task_id, task in enumerate(tasks):
        task["worker_id"] = task_id % NUM_WORKERS

    ok_count = 0
    fail_count = 0
    json_item_count = 0
    duplicate_count = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_single_clip, task)
            for task in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing clips",
        ):
            result = future.result()

            if result["ok"]:
                ok_count += 1

                for meta_item in result["meta_items"]:
                    key = build_unique_key(meta_item)

                    if ENABLE_APPEND_MODE and key in existing_keys:
                        duplicate_count += 1
                        continue

                    append_jsonl(SAVE_JSONL_PATH, meta_item)
                    existing_keys.add(key)
                    json_item_count += 1

            else:
                fail_count += 1
                if fail_count <= 50:
                    print(f"⚠️ failed: {result['err']}")

    num_items = jsonl_to_json(
        jsonl_path=SAVE_JSONL_PATH,
        json_path=SAVE_JSON_PATH,
    )

    print("")
    print("🎉 Done")
    print(f"✅ save root:              {SAVE_ROOT}")
    print(f"✅ rgb video root:         {RGB_VIDEO_ROOT}")
    print(f"✅ sobel edge video root:  {SOBEL_VIDEO_ROOT}")
    print(f"✅ da2 depth video root:   {DA2_DEPTH_VIDEO_ROOT}")
    print(f"✅ jsonl:                  {SAVE_JSONL_PATH}")
    print(f"✅ json:                   {SAVE_JSON_PATH}")
    print(f"✅ scanned valid episodes: {processed_episodes}")
    print(f"✅ scan skipped episodes:  {skipped_episodes_scan}")
    print(f"⏭️  skipped (exist):        {skipped_clips_exist}")
    print(f"✅ ok clips:               {ok_count}")
    print(f"✅ failed clips:           {fail_count}")
    print(f"✅ new json items:         {json_item_count}")
    print(f"✅ duplicate skipped:      {duplicate_count}")
    print(f"✅ total json items:       {num_items}")
    print(f"✅ output size:            {OUT_W}x{OUT_H}")


if __name__ == "__main__":
    main()