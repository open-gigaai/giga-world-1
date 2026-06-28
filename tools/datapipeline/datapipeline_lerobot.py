import os
import sys
import argparse
from pathlib import Path

# Add the project root to sys.path so thirdparty/ and tools/ can be imported.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def resolve_path(path):
    """Resolve relative paths against the project root and keep absolute paths unchanged."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p.resolve())


def resolve_path_list(paths):
    return [resolve_path(p) for p in paths]

# Force line-buffered stdout/stderr so subprocess logs are flushed in real time.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.multiprocessing as mp

import numpy as np
import pandas as pd
import time
import av
import cv2
import pickle
import json
from PIL import Image
from thirdparty.depth_abything_v2.pipeline_da2 import get_depth_image_batch_da2, get_image_depth_anything

from tools.image_utils import load_video_frames, save_frames

# ===================== Config (Can Modify) =====================
VIDEO_FPS = 30
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
TARGET_HEIGHT = 480           # Forced output height.
TARGET_WIDTH = 640            # Forced output width.

# Qwen3-VL captioning configuration
CAPTION_MODEL_PATH = "/shared_disk/models/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/"
CAPTION_MAX_PIXELS = 360 * 420
CAPTION_FPS = 2.0
CAPTION_MAX_NEW_TOKENS = 256

INSTRUCTION_TEMPLATE = """You are an expert in Embodied AI. Carefully watch the provided video and describe the robot's actual behavior in the scene.

Task hint: {task}

Requirements:
- The task hint describes the complete task, while the provided video may only show a subsegment of the full task execution. Your description should be based on the actual visual content of the video, rather than relying solely on the task hint.
- Describe the robot's appearance, the environment, the relevant objects, and the action sequence.
- Provide more fine-grained descriptions of background and foreground objects, including their positions, shapes, colors, materials, states, and spatial relationships to the robot.
- Be specific and precise. Clearly state which objects the robot interacts with and how it interacts with them.
- Keep it under 150 words.

Dense Recaption:"""

LONG_PROMPT_SEGMENT_FRAMES = 300  # Number of frames per captioning segment.
# =================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="LeRobot data processing pipeline (multi-GPU, per-task).")
    parser.add_argument(
        "--output_base",
        type=str,
        default="output",
        help="Base output directory (relative to project root, or absolute path)."
    )
    parser.add_argument(
        "--data_dir_list",
        type=str,
        nargs="+",
        default=[
            "origin_data",
        ],
        help="Input data directories (relative to project root, or absolute paths)."
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for parallel processing."
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=-1,
        help="Limit total number of episodes to process (for debug). -1 means no limit."
    )
    return parser.parse_args()


def resolve_data_dirs(data_dir_list):
    """
    If a parent directory is provided, automatically expand its task subdirectories.
    Only keep valid task folders that contain both data/ and videos/.
    """
    resolved = []
    for d in data_dir_list:
        if not os.path.isdir(d):
            print(f"Warning: {d} is not a directory, skip.")
            continue

        if os.path.isdir(os.path.join(d, "data")) and os.path.isdir(os.path.join(d, "videos")):
            resolved.append(d)
            continue

        for sub in sorted(os.listdir(d)):
            sub_path = os.path.join(d, sub)
            if not os.path.isdir(sub_path):
                continue
            if os.path.isdir(os.path.join(sub_path, "data")) and os.path.isdir(os.path.join(sub_path, "videos")):
                resolved.append(sub_path)

    return resolved


def load_episode_prompts(data_dir):
    """
    Load per-episode metadata from meta/episodes.jsonl.
    Returns a dict: episode_index -> {"tasks": [...], "length": int}
    """
    episodes_path = os.path.join(data_dir, "meta", "episodes.jsonl")
    if not os.path.exists(episodes_path):
        return {}

    episode_prompts = {}
    try:
        with open(episodes_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ep_idx = obj.get("episode_index")
                if ep_idx is not None:
                    episode_prompts[ep_idx] = {
                        "tasks": obj.get("tasks", []),
                        "length": obj.get("length", 0),
                    }
    except Exception as e:
        print(f"Warning: failed to load episodes.jsonl: {e}")

    return episode_prompts


def load_caption_model(device):
    """Load the Qwen3-VL caption model on the specified GPU and return (model, processor)."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    print(f"  Loading Qwen3-VL caption model on {device}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = Qwen3VLProcessor.from_pretrained(CAPTION_MODEL_PATH)
    print(f"  Qwen3-VL caption model loaded on {device}!")
    return model, processor


def _write_temp_video(frames_np, fps=VIDEO_FPS):
    """Write a numpy frame array to a temporary mp4 file and return its path. The caller is responsible for cleanup."""
    import tempfile
    h, w = frames_np[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    writer = cv2.VideoWriter(tmp_path, VIDEO_CODEC, fps, (w, h))
    for frame in frames_np:
        writer.write(cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_RGB2BGR))
    writer.release()
    return tmp_path


def _caption_segment_from_frames(caption_model, caption_processor, frames_np, task_hint, device):
    """Write a group of frames into a temporary clip and send it to Qwen3-VL for captioning. Return an empty string on failure."""
    from qwen_vl_utils import process_vision_info

    tmp_path = None
    try:
        tmp_path = _write_temp_video(frames_np)
        prompt_text = INSTRUCTION_TEMPLATE.format(task=task_hint)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": tmp_path,
                        "max_pixels": CAPTION_MAX_PIXELS,
                        "fps": CAPTION_FPS,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = caption_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = caption_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = caption_model.generate(**inputs, max_new_tokens=CAPTION_MAX_NEW_TOKENS)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = caption_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return caption.strip()
    except Exception as e:
        print(f"    Warning: failed to generate caption for segment: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_long_prompt(caption_model, caption_processor, cam_high_frames, task_hint, device):
    """
    Split cam_high frames into segments of LONG_PROMPT_SEGMENT_FRAMES frames,
    then call Qwen3-VL on each segment to generate dense captions.
    Returns a dict: {"long prompt 1": {"start_idx": "0", "end_idx": "299", "caption": "..."}, ...}
    """
    total_frames = len(cam_high_frames)
    result = {}
    seg_idx = 1
    for start in range(0, total_frames, LONG_PROMPT_SEGMENT_FRAMES):
        end = min(start + LONG_PROMPT_SEGMENT_FRAMES - 1, total_frames - 1)
        caption = _caption_segment_from_frames(
            caption_model, caption_processor,
            cam_high_frames[start: end + 1], task_hint, device
        )
        result[f"long prompt {seg_idx}"] = {
            "start_idx": str(start),
            "end_idx": str(end),
            "caption": caption,
        }
        seg_idx += 1
    return result


def is_episode_processed(output_dir, episode_name):
    """Check whether an episode has already been fully processed (all gt + depth videos exist)."""
    required_files = [
        f"{output_dir}/gt/cam_high/{episode_name}.mp4",
        f"{output_dir}/gt/cam_left_wrist/{episode_name}.mp4",
        f"{output_dir}/gt/cam_right_wrist/{episode_name}.mp4",
        f"{output_dir}/depth/cam_high/{episode_name}.mp4",
        f"{output_dir}/depth/cam_left_wrist/{episode_name}.mp4",
        f"{output_dir}/depth/cam_right_wrist/{episode_name}.mp4",
    ]
    return all(os.path.exists(f) for f in required_files)


def load_caption_cache(output_dir):
    """Load the caption cache file and return a dict: episode_name -> long_prompt dict."""
    cache_path = os.path.join(output_dir, "labels", "caption_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_caption_cache(output_dir, cache):
    """Write the caption cache back to disk atomically (write temp file first, then replace)."""
    cache_path = os.path.join(output_dir, "labels", "caption_cache.json")
    tmp_path = cache_path + ".tmp"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(cache, f)
    os.replace(tmp_path, cache_path)


def collect_tasks_for_one_dir(data_dir, output_base):
    """Collect the full episode task list for a single data_dir."""
    tasks = []
    sub_name = os.path.basename(data_dir)
    output_dir = os.path.join(output_base, sub_name)
    chunk_root = os.path.join(data_dir, "data")

    if not os.path.isdir(chunk_root):
        return tasks

    for chunk in sorted(os.listdir(chunk_root)):
        chunk_path = os.path.join(chunk_root, chunk)
        if not os.path.isdir(chunk_path):
            continue
        for ep in sorted(os.listdir(chunk_path)):
            if not ep.endswith(".parquet"):
                continue
            ep_path = os.path.join(chunk_path, ep)
            vid_dir = os.path.join(data_dir, "videos", chunk)
            tasks.append({
                "data_dir": data_dir,
                "sub_name": sub_name,
                "output_dir": output_dir,
                "ep_path": ep_path,
                "vid_dir": vid_dir,
                "ep_filename": ep,
            })
    return tasks


def process_episode(task, episode_prompts, caption_model, caption_processor, depth_model, depth_processor, device, caption_cache):
    """
    Process a single episode. Returns (status, episode_name, meta, error_msg)
    where status is one of {"done", "skipped", "error"}.

    caption_cache is a shared dict (episode_name -> long_prompt) used to reuse
    caption results across reruns.
    """
    output_dir = task["output_dir"]
    ep_path = task["ep_path"]
    vid_dir = task["vid_dir"]
    ep_filename = task["ep_filename"]
    sub_name = task["sub_name"]

    gt_cam_right_wrist_video_path = os.path.join(
        vid_dir, "observation.images.cam_right_wrist", ep_filename.replace(".parquet", ".mp4")
    )
    gt_cam_left_wrist_video_path = os.path.join(
        vid_dir, "observation.images.cam_left_wrist", ep_filename.replace(".parquet", ".mp4")
    )
    gt_cam_high_video_path = os.path.join(
        vid_dir, "observation.images.cam_high", ep_filename.replace(".parquet", ".mp4")
    )

    df = pd.read_parquet(ep_path)
    action = np.array(df['action'].tolist(), dtype=np.float32)
    observation = np.array(df['observation.state'].tolist(), dtype=np.float32)
    episode_index = df['episode_index'].values[0]
    episode_name = f"episode_{episode_index:06d}"
    total_frames = len(observation)

    # ========== short-prompt: read tasks from episodes.jsonl and build a structured dict ==========
    ep_info = episode_prompts.get(episode_index, {})
    ep_tasks = ep_info.get("tasks", []) if isinstance(ep_info, dict) else []
    if not ep_tasks:
        ep_tasks = [sub_name]

    short_prompt = {}
    for t_idx, task_desc in enumerate(ep_tasks, start=1):
        short_prompt[f"task{t_idx}"] = {
            "start_idx": "0",
            "end_idx": str(total_frames - 1),
            "description": task_desc,
        }
    task_hint = ep_tasks[0]

    video_already_done = is_episode_processed(output_dir, episode_name)

    # ========== long-prompt: prefer cache; otherwise run Qwen3-VL captioning ==========
    if episode_name in caption_cache:
        long_prompt = caption_cache[episode_name]
    else:
        # Video frames are required before caption generation can run.
        try:
            cam_high_frames = load_video_frames(gt_cam_high_video_path)
        except Exception as e:
            return ("error", episode_name, None, f"gt_video(caption): {str(e)}")

        long_prompt = generate_long_prompt(
            caption_model, caption_processor,
            cam_high_frames, task_hint, device
        )
        caption_cache[episode_name] = long_prompt
        save_caption_cache(output_dir, caption_cache)

    meta = {
        "action": action.tolist(),
        "data_index": -1,
        "episode_name": episode_name,
        "cam_high_video_path": f"{output_dir}/gt/cam_high/{episode_name}.mp4",
        "cam_left_wrist_video_path": f"{output_dir}/gt/cam_left_wrist/{episode_name}.mp4",
        "cam_right_wrist_video_path": f"{output_dir}/gt/cam_right_wrist/{episode_name}.mp4",
        "cam_high_depth_path": f"{output_dir}/depth/cam_high/{episode_name}.mp4",
        "cam_left_wrist_depth_path": f"{output_dir}/depth/cam_left_wrist/{episode_name}.mp4",
        "cam_right_wrist_depth_path": f"{output_dir}/depth/cam_right_wrist/{episode_name}.mp4",
        "qpos": observation.tolist(),
        "video_height": TARGET_HEIGHT,
        "video_width": TARGET_WIDTH,
        "video_length": total_frames,
        "short-prompt": short_prompt,
        "long-prompt": long_prompt,
    }

    # Skip video writing if the episode has already been processed.
    if video_already_done:
        return ("skipped", episode_name, meta, None)

    try:
        gt_frames = {
            "right_camera": load_video_frames(gt_cam_right_wrist_video_path),
            "left_camera": load_video_frames(gt_cam_left_wrist_video_path),
            "head_camera": load_video_frames(gt_cam_high_video_path),
        }
    except av.error.InvalidDataError as e:
        return ("error", episode_name, None, f"gt_video: {str(e)}")
    except Exception as e:
        return ("error", episode_name, None, f"gt_video: {str(e)}")

    save_frames(gt_frames, f"{output_dir}/gt", episode_name, VIDEO_FPS, None, TARGET_WIDTH, TARGET_HEIGHT)

    try:
        BATCH_SIZE = 32
        gt_depth_frames = {}
        for cam_key, np_frames in gt_frames.items():
            all_depth_np = []
            for i in range(0, len(np_frames), BATCH_SIZE):
                batch_np = np_frames[i:i+BATCH_SIZE]
                pil_batch = [Image.fromarray(f.astype('uint8')) for f in batch_np]
                depth_pil_batch = get_depth_image_batch_da2(depth_model, depth_processor, pil_batch, device=device)
                all_depth_np.extend([np.array(d) for d in depth_pil_batch])
            gt_depth_frames[cam_key] = np.stack(all_depth_np)
        save_frames(gt_depth_frames, f"{output_dir}/depth", episode_name, VIDEO_FPS, None, TARGET_WIDTH, TARGET_HEIGHT)
    except Exception as e:
        return ("error", episode_name, None, f"depth: {str(e)}")

    return ("done", episode_name, meta, None)


def worker(rank, world_size, task_batch, episode_prompts, output_base, total_in_rank):
    """
    Per-task GPU worker: load models and process the episodes assigned to this rank.
    Each worker handles only one task at a time, writes temporary results immediately,
    and exits so GPU memory can be released.
    """
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    depth_model, depth_processor = get_image_depth_anything(device, type="da2_small")
    print(f"[Rank {rank}] Depth Anything v2 model loaded on {device}!", flush=True)

    caption_model, caption_processor = load_caption_model(device)
    print(f"[Rank {rank}] Qwen3-VL caption model loaded on {device}!", flush=True)

    my_tasks = task_batch[rank::world_size]
    print(f"[Rank {rank}] Processing {len(my_tasks)} episodes", flush=True)

    # Each rank maintains its own caption cache scoped by output_dir.
    # Multiple ranks may write the same file concurrently, but atomic replacement
    # prevents corruption; whichever rank writes last still produces a valid cache.
    caption_cache = load_caption_cache(my_tasks[0]["output_dir"]) if my_tasks else {}
    print(f"[Rank {rank}] Loaded {len(caption_cache)} cached captions", flush=True)

    results = []
    start_time = time.time()
    for i, task in enumerate(my_tasks):
        ep_start = time.time()
        try:
            status, episode_name, meta, err = process_episode(
                task, episode_prompts, caption_model, caption_processor,
                depth_model, depth_processor, device, caption_cache
            )
        except Exception as e:
            status, episode_name, meta, err = ("error", task["ep_filename"], None, f"unexpected: {str(e)}")

        results.append({
            "sub_name": task["sub_name"],
            "status": status,
            "episode_name": episode_name,
            "meta": meta,
            "error": err,
        })

        ep_elapsed = time.time() - ep_start
        total_elapsed = time.time() - start_time
        done_count = i + 1
        avg_per_ep = total_elapsed / done_count
        eta = avg_per_ep * (len(my_tasks) - done_count)

        def fmt(s):
            return f"{int(s//3600)}h {int(s%3600//60)}m {int(s%60)}s"

        if status == "done":
            print(f"[Rank {rank}] Done    [{done_count}/{len(my_tasks)} {done_count/len(my_tasks)*100:.1f}%] "
                  f"{episode_name} | ep={fmt(ep_elapsed)} | total={fmt(total_elapsed)} | ETA={fmt(eta)}", flush=True)
        elif status == "skipped":
            print(f"[Rank {rank}] Skipped [{done_count}/{len(my_tasks)} {done_count/len(my_tasks)*100:.1f}%] "
                  f"{episode_name}", flush=True)
        else:
            print(f"[Rank {rank}] Error   [{done_count}/{len(my_tasks)} {done_count/len(my_tasks)*100:.1f}%] "
                  f"{episode_name} - {err}", flush=True)

    rank_result_path = os.path.join(output_base, f".rank_{rank}_results.pkl")
    with open(rank_result_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[Rank {rank}] Finished, saved {len(results)} results to {rank_result_path}", flush=True)
    torch.cuda.empty_cache()


def aggregate_and_save(data_dir, output_base, world_size):
    """
    Merge temporary results from all ranks for the current task, write labels,
    and then delete the temporary files.
    """
    sub_name = os.path.basename(data_dir)
    output_dir = os.path.join(output_base, sub_name)

    all_results = []
    for rank in range(world_size):
        rank_result_path = os.path.join(output_base, f".rank_{rank}_results.pkl")
        if os.path.exists(rank_result_path):
            with open(rank_result_path, "rb") as f:
                all_results.extend(pickle.load(f))

    # Sort by episode_name to keep data_index stable across runs.
    all_results.sort(key=lambda r: r["episode_name"])

    dataset_meta = []
    skipped_episodes = []
    error_episodes = []
    for r in all_results:
        if r["status"] in ("done", "skipped") and r["meta"] is not None:
            meta = dict(r["meta"])
            meta["data_index"] = len(dataset_meta)
            dataset_meta.append(meta)
            if r["status"] == "skipped":
                skipped_episodes.append(r["episode_name"])
        elif r["status"] == "error":
            error_episodes.append((r["episode_name"], r["error"]))

    # Write label files.
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    with open(f"{output_dir}/labels/data.pkl", "wb") as f:
        pickle.dump(dataset_meta, f)

    config = {
        "_class_name": "PklDataset",
        "_key_names": list(dataset_meta[0].keys()) if dataset_meta else [],
        "data_size": len(dataset_meta)
    }
    with open(f"{output_dir}/labels/config.json", "w") as f:
        json.dump(config, f, indent=4)

    dataset_config = {
        "_class_name": "Dataset",
        "config_paths": ["labels/config.json"]
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(dataset_config, f, indent=4)

    # Clean up temporary files.
    for rank in range(world_size):
        rank_result_path = os.path.join(output_base, f".rank_{rank}_results.pkl")
        if os.path.exists(rank_result_path):
            os.remove(rank_result_path)
    # Delete the caption cache after labels are written; it will be regenerated on the next rerun of this task.
    caption_cache_path = os.path.join(output_dir, "labels", "caption_cache.json")
    if os.path.exists(caption_cache_path):
        os.remove(caption_cache_path)

    print(f"\n{'='*60}")
    print(f"Task done: {sub_name}")
    print(f"   Output dir: {output_dir}")
    print(f"   Successfully processed: {len(dataset_meta)} episodes")
    print(f"   Skipped (already processed): {len(skipped_episodes)} episodes")
    if skipped_episodes:
        for ep in skipped_episodes:
            print(f"      - {ep}")
    print(f"   Error skipped: {len(error_episodes)} episodes")
    if error_episodes:
        for ep, err in error_episodes:
            print(f"      - {ep}: {err}")
    print(f"{'='*60}\n")


# ===================== Main =====================
if __name__ == "__main__":
    args = parse_args()
    OUTPUT_BASE = resolve_path(args.output_base)
    DATA_DIR_LIST = resolve_data_dirs(resolve_path_list(args.data_dir_list))

    if not DATA_DIR_LIST:
        print("No valid data directories found, exit.")
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No CUDA GPUs available, exit.")
        sys.exit(1)

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Found {len(DATA_DIR_LIST)} task(s) to process, using up to {args.num_gpus} GPUs")
    print(f"{'='*60}\n")

    # Process tasks one by one: each task runs in its own mp.spawn round, then labels are written immediately and memory is released.
    for task_idx, data_dir in enumerate(DATA_DIR_LIST):
        sub_name = os.path.basename(data_dir)
        output_dir = os.path.join(OUTPUT_BASE, sub_name)

        # If data.pkl already exists, this task has already been fully processed and can be skipped.
        data_pkl_path = os.path.join(output_dir, "labels", "data.pkl")
        if os.path.exists(data_pkl_path):
            print(f"\n[Task {task_idx+1}/{len(DATA_DIR_LIST)}] {sub_name} — already done (data.pkl exists), skip.")
            continue

        for d in [f"{output_dir}/gt", f"{output_dir}/depth", f"{output_dir}/labels"]:
            os.makedirs(d, exist_ok=True)

        episode_prompts = load_episode_prompts(data_dir)
        task_batch = collect_tasks_for_one_dir(data_dir, OUTPUT_BASE)

        if args.max_tasks > 0 and len(task_batch) > args.max_tasks:
            print(f"[DEBUG] Limiting episodes: {len(task_batch)} -> {args.max_tasks}")
            task_batch = task_batch[:args.max_tasks]

        print(f"\n{'='*60}")
        print(f"[Task {task_idx+1}/{len(DATA_DIR_LIST)}] {sub_name}")
        print(f"   Episodes: {len(task_batch)}  |  Prompts loaded: {len(episode_prompts)}")
        print(f"   Source : {data_dir}")
        print(f"   Output : {output_dir}")
        print(f"{'='*60}\n")

        if len(task_batch) == 0:
            print("No episodes found, skip.\n")
            continue

        world_size = min(args.num_gpus, available_gpus, len(task_batch))
        if world_size < args.num_gpus:
            print(f"Warning: using {world_size} GPUs "
                  f"(requested={args.num_gpus}, available={available_gpus}, episodes={len(task_batch)})")

        # Launch workers for the episodes of the current task only.
        mp.spawn(
            worker,
            args=(world_size, task_batch, episode_prompts, OUTPUT_BASE, len(task_batch)),
            nprocs=world_size,
            join=True,
        )

        # Aggregate results, write labels, and clean temporary files as soon as all ranks finish.
        aggregate_and_save(data_dir, OUTPUT_BASE, world_size)

    print("\n" + "="*30)
    print("All tasks completed!")
    print("="*30 + "\n")
