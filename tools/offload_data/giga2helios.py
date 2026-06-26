import os
import json
import pickle
from tqdm import tqdm


DATA_ROOTS = [
    "/shared_disk/users/hao.li/giga_world_1_data/agibot/lerobot/agibot_g1_task_356",
    "/shared_disk/users/hao.li/giga_world_1_data/agibot/lerobot/agibot_g1_task_357",
    "/shared_disk/users/hao.li/giga_world_1_data/agibot/lerobot/agibot_g1_task_358",
    "/shared_disk/users/hao.li/giga_world_1_data/agibot/lerobot/agibot_g1_task_359",
    "/shared_disk/users/hao.li/giga_world_1_data/agibot/lerobot/agibot_g1_task_360",
    "/shared_disk/users/hao.li/giga_world_1_data/agibot/lerobot/agibot_g1_task_361",
]

# 原始 agibot 数据根目录
SOURCE_VIDEO_ROOT = "/shared_disk/users/hao.li/giga_world_1_data/agibot/lerobot"

# Helios 输出目录
SAVE_ROOT = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/agibot_debug"
VIDEO_ROOT = os.path.join(SAVE_ROOT, "videos")

SAVE_JSONL_PATH = os.path.join(SAVE_ROOT, "helios_front_dataset.jsonl")
SAVE_JSON_PATH = os.path.join(SAVE_ROOT, "helios_front_dataset.json")

NUM_FRAMES = 129
STRIDE = 129
T_DOWNSAMPLE = 1

FPS = 24.0
MAX_EPISODES = None   # None = 全部处理

RESET_JSONL = True
RESET_VIDEO_LINKS = False

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(VIDEO_ROOT, exist_ok=True)

if RESET_JSONL and os.path.exists(SAVE_JSONL_PATH):
    os.remove(SAVE_JSONL_PATH)


def get_front_video_path(item):
    return (
        item.get("cam_high_video_path", None)
        or item.get("front_video_path", None)
    )


def parse_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def choose_caption_from_long_prompt(item, clip_start, clip_end):
    long_prompt = item.get("long-prompt", None)

    if isinstance(long_prompt, dict) and len(long_prompt) > 0:
        best_caption = None
        best_overlap = -1
        best_center_dist = float("inf")

        clip_center = (clip_start + clip_end - 1) / 2.0

        for _, seg in long_prompt.items():
            seg_start = parse_int(seg.get("start_idx", 0))
            seg_end = parse_int(seg.get("end_idx", seg_start))
            caption = seg.get("caption", "")

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
                best_caption = caption

        if best_caption:
            return best_caption

    if item.get("prompt", ""):
        return item["prompt"]

    short_prompt = item.get("short-prompt", None)
    if isinstance(short_prompt, dict) and len(short_prompt) > 0:
        first_key = list(short_prompt.keys())[0]
        return short_prompt[first_key].get("description", "")

    return ""


def build_clip_starts(video_length, required_raw_frames, stride):
    return list(range(0, video_length - required_raw_frames + 1, stride))


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
            if not line:
                continue
            data.append(json.loads(line))

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return len(data)


def ensure_video_link(src_video_path):
    """
    在 VIDEO_ROOT 下创建和 SOURCE_VIDEO_ROOT 相同的相对路径结构。
    返回写入 json 的相对 path。
    """
    src_video_path = os.path.abspath(src_video_path)
    source_root = os.path.abspath(SOURCE_VIDEO_ROOT)

    rel_path = os.path.relpath(src_video_path, source_root)
    dst_video_path = os.path.join(VIDEO_ROOT, rel_path)

    os.makedirs(os.path.dirname(dst_video_path), exist_ok=True)

    if os.path.exists(dst_video_path):
        return rel_path

    try:
        # 优先硬链接：不占额外空间，速度最快
        os.link(src_video_path, dst_video_path)
    except OSError:
        # 如果跨文件系统硬链接失败，则退化为软链接
        os.symlink(src_video_path, dst_video_path)

    return rel_path


def main():
    processed_episodes = 0
    skipped_episodes = 0
    total_clips = 0

    required_raw_frames = (NUM_FRAMES - 1) * T_DOWNSAMPLE + 1

    for root in DATA_ROOTS:
        if MAX_EPISODES is not None and processed_episodes >= MAX_EPISODES:
            break

        pkl_path = os.path.join(root, "labels", "data.pkl")

        if not os.path.exists(pkl_path):
            print(f"⚠️ skip missing: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)

        root_name = os.path.basename(root)

        for item_id, item in enumerate(tqdm(meta, desc=root_name)):
            if MAX_EPISODES is not None and processed_episodes >= MAX_EPISODES:
                break

            src_video_path = get_front_video_path(item)

            if src_video_path is None or not os.path.exists(src_video_path):
                skipped_episodes += 1
                continue

            video_length = item.get("video_length", None)
            video_height = item.get("video_height", None)
            video_width = item.get("video_width", None)

            if video_length is None or video_height is None or video_width is None:
                skipped_episodes += 1
                continue

            video_length = int(video_length)
            video_height = int(video_height)
            video_width = int(video_width)

            if video_length < required_raw_frames:
                skipped_episodes += 1
                continue

            clip_starts = build_clip_starts(
                video_length=video_length,
                required_raw_frames=required_raw_frames,
                stride=STRIDE,
            )

            if len(clip_starts) == 0:
                skipped_episodes += 1
                continue

            try:
                rel_video_path = ensure_video_link(src_video_path)
            except Exception as e:
                print(f"⚠️ link failed: {src_video_path}, error={e}")
                skipped_episodes += 1
                continue

            processed_episodes += 1

            episode_name = item.get("episode_name", f"{item_id:06d}")

            for clip_id, start_idx in enumerate(clip_starts):
                raw_end_idx = start_idx + required_raw_frames

                caption = choose_caption_from_long_prompt(
                    item=item,
                    clip_start=start_idx,
                    clip_end=raw_end_idx,
                )

                meta_item = {
                    "cut": [start_idx, raw_end_idx],
                    "crop": [0, video_width, 0, video_height],
                    "fps": FPS,
                    "num_frames": NUM_FRAMES,
                    "resolution": {
                        "height": video_height,
                        "width": video_width,
                    },
                    "cap": [caption],
                    "path": rel_video_path,

                    # debug 字段，不需要可以删
                    "source_video": os.path.abspath(src_video_path),
                    "source_cut": [start_idx, raw_end_idx],
                    "episode_name": episode_name,
                    "clip_id": clip_id,
                }

                append_jsonl(SAVE_JSONL_PATH, meta_item)
                total_clips += 1

    num_json_items = jsonl_to_json(
        jsonl_path=SAVE_JSONL_PATH,
        json_path=SAVE_JSON_PATH,
    )

    print(f"✅ saved jsonl: {SAVE_JSONL_PATH}")
    print(f"✅ saved json:  {SAVE_JSON_PATH}")
    print(f"✅ video root:   {VIDEO_ROOT}")
    print(f"✅ processed episodes: {processed_episodes}")
    print(f"✅ skipped episodes: {skipped_episodes}")
    print(f"✅ total clips: {total_clips}")
    print(f"✅ json items: {num_json_items}")


if __name__ == "__main__":
    main()