import argparse
import base64
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# Percentage of frames to use for start/end portions
DRIFT_RATIO = 0.15

PROMPT_VIDEO_DETECTION = """
Your task is to analyze ##NUM_FRAMES## frames sampled across a long video (e.g., ~1500 frames) to determine if it is AI-generated.

### CONTEXT:
Frames are sampled at wide intervals. Scene changes or camera movements are expected. Focus on individual frame integrity and local physical logic rather than global consistency.

### EVALUATION CRITERIA:
1. **Single-Frame Technical Flaws**: Look for AI-specific rendering "hallucinations":
   - **Textures**: "Melting" surfaces, plastic-like skin, or chaotic patterns in complex areas (e.g., water ripples, foliage, fire).
   - **Edges**: Unnatural blurring or "auras" around moving subjects where they meet the background.
2. **Local Physical Logic**: Within any given frame or small cluster of frames:
   - Do shadows and reflections align with the visible light sources?
   - Are objects interacting naturally with their environment (e.g., feet touching the ground properly, hands grasping objects correctly)?
3. **Biological Anomalies**: If humans appear, inspect for:
   - Anatomical errors: Extra fingers, asymmetric eyes, or "fused" teeth.
   - Unnatural micro-expressions or "dead" eyes lacking specular highlights.
4. **Transient Artifacts**: Even in sparse samples, look for "ghosting" or objects that seem to be partially transparent or merging with other objects (common in AI diffusion).

### SCORING SCALE:
- 1 (Definitely AI): Clear anatomical deformities, "melting" textures, or impossible physical interactions.
- 2 (Likely AI): Presence of "uncanny valley" effects, suspicious texture smoothing, or minor physical illogic.
- 3 (Uncertain): Ambiguous; could be low-quality real-world footage, heavy motion blur, or high-end AI.
- 4 (Likely Real): Consistent organic details, natural motion blur, and logical lighting.
- 5 (Definitely Real): Perfect high-frequency details (pores, fabric, grain), flawless physics, and natural anatomy.

### OUTPUT INSTRUCTION:
Return ONLY the integer score (1-5). No explanation.
""".strip()


def image_to_base64(image):
    """Convert image to base64 string"""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def resize_long_side(image, target_long=512):
    """Resize image keeping aspect ratio"""
    h, w = image.shape[:2]
    if h >= w:
        new_h = target_long
        new_w = int(w * target_long / h)
    else:
        new_w = target_long
        new_h = int(h * target_long / w)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames_from_portion(video_path, num_frames=8, portion="start"):
    """
    Extract frames from start or end portion of video
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        portion: 'start' or 'end'
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate portion boundaries
    num_drift_frames = max(1, int(total_frames * DRIFT_RATIO))

    if portion == "start":
        start_frame = 0
        end_frame = num_drift_frames
    else:  # end
        start_frame = total_frames - num_drift_frames
        end_frame = total_frames

    # Extract frames evenly from the portion
    frame_interval = max((end_frame - start_frame) // num_frames, 1)
    frames = []

    for i in range(num_frames):
        frame_idx = start_frame + i * frame_interval
        if frame_idx >= end_frame:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            resized = resize_long_side(frame, 512)
            frames.append(resized)

    cap.release()
    return frames


def call_gpt(image_frames_base64, model_name, api_key, base_url, num_frames=8, temperature=0.0):
    """Call GPT API to evaluate video naturalness"""
    client = OpenAI(api_key=api_key, base_url=base_url)

    content_list = []
    for frame in image_frames_base64:
        content_list.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
        )
    content_list.append({"type": "text", "text": PROMPT_VIDEO_DETECTION.replace("##NUM_FRAMES##", str(num_frames))})

    response = client.chat.completions.create(
        model=model_name,
        stream=False,
        temperature=temperature,
        messages=[{"role": "user", "content": content_list}],
    )

    score = response.choices[0].message.content.strip()
    return score


def evaluate_portion_naturalness(video_path, api_key, model_name, base_url, num_frames=8, portion="start"):
    """Evaluate naturalness for start or end portion of video"""
    try:
        frames = extract_frames_from_portion(video_path, num_frames, portion)
        frames_base64 = [image_to_base64(f) for f in frames]
        score_str = call_gpt(frames_base64, model_name, api_key, base_url, num_frames)

        # Parse score
        score = float(score_str)
        score = max(1.0, min(5.0, score))
        # Normalize to [0, 1] (1-5 to 0-1)
        score = (score - 1) / 4.0

        return score, score_str
    except Exception as e:
        print(f"Error evaluating video portion {portion}: {str(e)}")
        raise


def evaluate_drifting_naturalness(video_path, api_key, model_name, base_url, num_frames=8):
    """
    Evaluate drifting naturalness for a single video
    Returns: (drift_score, start_score, end_score, start_raw, end_raw)
    """
    # Evaluate start portion
    start_score, start_raw = evaluate_portion_naturalness(
        video_path, api_key, model_name, base_url, num_frames, portion="start"
    )

    # Evaluate end portion
    end_score, end_raw = evaluate_portion_naturalness(
        video_path, api_key, model_name, base_url, num_frames, portion="end"
    )

    # Calculate drift as absolute difference
    drift_score = abs(start_score - end_score)

    return drift_score, start_score, end_score, start_raw, end_raw


def process_video_worker(args_tuple):
    """Worker function for parallel processing"""
    video_path, video_id, video_name, api_key, model_name, base_url, num_frames = args_tuple
    try:
        drift_score, start_score, end_score, start_raw, end_raw = evaluate_drifting_naturalness(
            video_path, api_key, model_name, base_url, num_frames
        )
        return {
            "id": video_id,
            "video_name": video_name,
            "drift_naturalness_score": drift_score,
            "start_naturalness_score": start_score,
            "end_naturalness_score": end_score,
            "start_raw_score": start_raw,
            "end_raw_score": end_raw,
        }
    except Exception as e:
        print(f"Error processing {video_name}: {str(e)}")
        return None


def main(args):
    baseline_name = os.path.basename(args.video_dir)
    output_path = os.path.join(args.output_path, baseline_name)
    output_json_path = os.path.join(output_path, "drifting_naturalness_results.json")

    print(f"Using API: {args.base_url}")
    print(f"Model: {args.model_name}")

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

    # Get video files
    video_files = glob.glob(os.path.join(args.video_dir, "*_*_ori*.mp4"))
    video_files.sort(key=lambda x: int(re.search(r"(\d+)_", os.path.basename(x)).group(1)))
    print(f"\nFound {len(video_files)} videos in directory")

    # Check which videos need processing
    results = []
    drift_scores = []
    tasks = []

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
            drift_scores.append(existing_results[video_id]["drift_naturalness_score"])
        else:
            # Need to process
            tasks.append(
                (
                    video_path,
                    video_id,
                    video_name,
                    args.api_key,
                    args.model_name,
                    args.base_url,
                    args.num_frames,
                )
            )

    print(f"Already processed: {len(existing_results)} videos")
    print(f"Need to process: {len(tasks)} videos")

    # Evaluate remaining videos in parallel
    if tasks:
        results_dict = {}

        print(f"Evaluating videos with {args.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_idx = {executor.submit(process_video_worker, task): idx for idx, task in enumerate(tasks)}

            for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="Evaluating"):
                idx = future_to_idx[future]
                result = future.result()
                if result is not None:
                    results_dict[idx] = result

        # Add new results in order
        new_results = [results_dict[i] for i in sorted(results_dict.keys())]
        results.extend(new_results)
        drift_scores.extend([r["drift_naturalness_score"] for r in new_results])
    else:
        print("No videos to process. Skipping evaluation.")
        return

    # Sort all results by video_id
    results_sorted = sorted(results, key=lambda x: x["id"])

    # Calculate overall metrics and save final results
    if drift_scores:
        avg_drift = sum(drift_scores) / len(drift_scores)

        output = {
            "metric": "drifting_naturalness",
            "description": f"Start-end contrast of naturalness (first/last {DRIFT_RATIO * 100:.0f}% frames)",
            "average_drift_score": avg_drift,
            "num_videos": len(drift_scores),
            "model_name": args.model_name,
            "per_video_results": results_sorted,
        }

        # Save results
        os.makedirs(output_path, exist_ok=True)

        with open(output_json_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n{'=' * 60}")
        print("Results Summary:")
        print(f"{'=' * 60}")
        print(f"Average Drifting Naturalness Score: {avg_drift:.4f}")
        print("(Lower is better - indicates less quality drift)")
        print(f"Number of videos evaluated: {len(drift_scores)}")
        print(f"Results saved to: {output_json_path}")
        print(f"{'=' * 60}\n")
    else:
        print("No videos were successfully evaluated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate drifting naturalness using VLM")

    # Input/Output arguments
    parser.add_argument("--input_csv", type=str, default="playground/gigaworld_t2v_prompts.csv")
    parser.add_argument("--video_dir", type=str, default="playground/toy-video")
    parser.add_argument("--output_path", type=str, default="playground/results")

    # API arguments
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-5.2-2025-12-11")
    parser.add_argument("--base_url", type=str, default=None)

    # Evaluation arguments
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=64)

    args = parser.parse_args()

    main(args)
