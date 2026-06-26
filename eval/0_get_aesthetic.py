import argparse
import glob
import json
import os
import re

import clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import clip_transform, load_video


BATCH_SIZE = 32


def get_aesthetic_model(path_to_model):
    """Load the aesthetic predictor model"""
    m = nn.Linear(768, 1)
    s = torch.load(path_to_model, map_location="cpu", weights_only=False)
    m.load_state_dict(s)
    m.eval()
    return m


def evaluate_aesthetic(aesthetic_model, clip_model, video_path, height=384, width=640, device="cuda"):
    """Evaluate aesthetic quality for a single video"""
    aesthetic_model.eval()
    clip_model.eval()

    # Load video frames
    images = load_video(video_path, height=height, width=width)
    image_transform = clip_transform(224)
    aesthetic_scores_list = []

    # Process in batches
    for i in range(0, len(images), BATCH_SIZE):
        image_batch = images[i : i + BATCH_SIZE]
        image_batch = image_transform(image_batch)
        image_batch = image_batch.to(device)

        with torch.no_grad():
            image_feats = clip_model.encode_image(image_batch).to(torch.float32)
            image_feats = F.normalize(image_feats, dim=-1, p=2)
            aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)

        aesthetic_scores_list.append(aesthetic_scores)

    # Combine all scores
    aesthetic_scores = torch.cat(aesthetic_scores_list, dim=0)
    normalized_aesthetic_scores = aesthetic_scores / 10.0
    avg_score = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)

    return avg_score.item()


def main(args):
    baseline_name = os.path.basename(args.video_dir)
    output_path = os.path.join(args.output_path, baseline_name)
    output_json_path = os.path.join(output_path, "aesthetic_results.json")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
            scores.append(existing_results[video_id]["aesthetic_score"])
        else:
            # Need to process
            videos_to_process.append((video_path, video_id, video_name))

    print(f"Already processed: {len(existing_results)} videos")
    print(f"Need to process: {len(videos_to_process)} videos")

    # Process remaining videos
    if videos_to_process:
        # Load models
        print("Loading CLIP model...")
        clip_model, preprocess = clip.load(args.clip_model_path, device=device)

        print("Loading aesthetic predictor model...")
        aesthetic_model = get_aesthetic_model(args.aesthetic_model_path).to(device)

        print("\nEvaluating remaining videos...")
        for video_path, video_id, video_name in tqdm(videos_to_process):
            try:
                score = evaluate_aesthetic(
                    aesthetic_model,
                    clip_model,
                    video_path,
                    height=args.height,
                    width=args.width,
                    device=device,
                )

                result_item = {"id": video_id, "video_name": video_name, "aesthetic_score": score}
                results.append(result_item)
                scores.append(score)

            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
                continue
    else:
        print("No videos to process. Skipping evaluation.")
        return

    # Calculate overall metrics
    if scores:
        avg_score = sum(scores) / len(scores)

        # Sort results by video_id
        results_sorted = sorted(results, key=lambda x: x["id"])

        output = {
            "metric": "aesthetic",
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
        print(f"Average Aesthetic Score: {avg_score:.4f}")
        print(f"Number of videos evaluated: {len(scores)}")
        print(f"Results saved to: {output_json_path}")
        print(f"{'=' * 60}\n")
    else:
        print("No videos were successfully evaluated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video aesthetic using CLIP + LAION aesthetic predictor")

    # Input/Output arguments
    parser.add_argument("--height", type=str, default=384)
    parser.add_argument("--width", type=str, default=640)
    parser.add_argument("--input_csv", type=str, default="playground/gigaworld_t2v_prompts.csv")
    parser.add_argument("--video_dir", type=str, default="playground/toy-video")
    parser.add_argument("--output_path", type=str, default="playground/results")

    # Model arguments
    parser.add_argument("--clip_model_path", type=str, default="checkpoints/aesthetic_model/ViT-L-14.pt")
    parser.add_argument(
        "--aesthetic_model_path", type=str, default="checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth"
    )

    args = parser.parse_args()

    main(args)
