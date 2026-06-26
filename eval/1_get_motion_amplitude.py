import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utils import load_video


def _downscale_maps(flow_maps, downscale_size=16):
    """Resize flow maps for score calculation"""
    downscaled = []
    for flow in flow_maps:
        h, w = flow.shape[:2]
        new_h = int(h * (downscale_size / w))
        downscaled.append(cv2.resize(flow, (downscale_size, new_h), interpolation=cv2.INTER_AREA))
    return downscaled


def _motion_score(maps_or_masks):
    """Calculate mean score from maps or masks"""
    if len(maps_or_masks) == 0:
        return 0.0
    average_map = np.mean(np.array(maps_or_masks), axis=0)
    return float(np.mean(average_map))


def compute_farneback_optical_flow(frames):
    """Compute dense optical flow using Farneback algorithm"""
    if len(frames) < 2:
        return []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    flow_maps = []

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow_map = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow_maps.append(flow_map)
        prev_gray = gray

    return flow_maps


def evaluate_motion(video_info, height=384, width=640):
    video_path, video_id, video_name = video_info
    try:
        images = load_video(video_path, height=height, width=width, return_tensor=False)
        farneback_maps = compute_farneback_optical_flow(images)
        score = _motion_score(_downscale_maps(farneback_maps))

        return {"id": video_id, "video_name": video_name, "motion_fb": abs(score), "success": True}
    except Exception as e:
        return {"id": video_id, "video_name": video_name, "error": str(e), "success": False}


def main(args):
    baseline_name = os.path.basename(args.video_dir)
    output_path = os.path.join(args.output_path, baseline_name)
    output_json_path = os.path.join(output_path, "motion_amplitude_results.json")

    # Load CSV file
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    df_dict = df.set_index("id").to_dict("index")

    # Validate CSV columns
    required_columns = ["id", "duration"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column. Found columns: {df.columns.tolist()}")

    # Load existing results if available
    existing_results = {}
    if os.path.exists(output_json_path):
        print(f"Found existing results at {output_json_path}, loading...")
        with open(output_json_path, "r") as f:
            existing_data = json.load(f)
            for item in existing_data.get("per_video_results", []):
                existing_results[item["id"]] = item
        print(f"Loaded {len(existing_results)} existing results")

    # Get all videos to process
    video_files = glob.glob(os.path.join(args.video_dir, "*_*_ori*.mp4"))
    video_files.sort(key=lambda x: int(re.search(r"(\d+)_", os.path.basename(x)).group(1)))
    print(f"\nFound {len(video_files)} videos in directory")

    # Check which videos need processing
    results = []
    scores = []
    videos_to_process = []

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        parts = video_name.replace(".mp4", "").split("_")
        video_id = int(parts[0])

        if video_id not in df_dict:
            print(f"Warning: Video {video_name} (id={video_id}) not found in CSV, skipping")
            continue

        # Check if already processed
        if video_id in existing_results:
            # Use existing result
            results.append(existing_results[video_id])
            scores.append(existing_results[video_id]["motion_fb"])
        else:
            # Need to process
            videos_to_process.append((video_path, video_id, video_name))

    print(f"Already processed: {len(existing_results)} videos")
    print(f"Need to process: {len(videos_to_process)} videos")

    # Process remaining videos
    if videos_to_process:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(evaluate_motion, v, args.height, args.width) for v in videos_to_process]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                res = future.result()
                if res["success"]:
                    results.append({"id": res["id"], "video_name": res["video_name"], "motion_fb": res["motion_fb"]})
                    scores.append(res["motion_fb"])
                else:
                    print(f"Error processing {res['video_name']}: {res.get('error')}")
    else:
        print("No videos to process. Skipping evaluation.")
        return

    # Calculate overall metrics
    if scores:
        avg_score = sum(scores) / len(scores)

        # Sort results by video_id
        results_sorted = sorted(results, key=lambda x: x["id"])

        output = {
            "metric": "motion_fb",
            "average_score": avg_score,
            "num_videos": len(scores),
            "per_video_results": results_sorted,
        }

        # Save results
        os.makedirs(output_path, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n{'=' * 60}")
        print("Results Summary:")
        print(f"{'=' * 60}")
        print(f"Average Motion Farneback Score:   {avg_score:.4f}")
        print(f"Number of videos evaluated: {len(scores)}")
        print(f"Results saved to: {output_json_path}")
        print(f"{'=' * 60}\n")
    else:
        print("No videos were successfully evaluated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video motion farneback")

    # Input/Output arguments
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--input_csv", type=str, default="playground/gigaworld_t2v_prompts.csv")
    parser.add_argument("--video_dir", type=str, default="playground/toy-video")
    parser.add_argument("--output_path", type=str, default="playground/results")

    # Evaluation arguments
    parser.add_argument("--num_workers", type=int, default=32)

    args = parser.parse_args()

    main(args)
