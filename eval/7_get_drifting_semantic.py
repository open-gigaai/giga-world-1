import argparse
import glob
import json
import os
import re

import pandas as pd
import torch
from tqdm import tqdm

from utils.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
from utils.third_party.ViCLIP.viclip import ViCLIP
from utils.utils import clip_transform, load_video_frames


# Percentage of frames to use for start/end portions
DRIFT_RATIO = 0.15


def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
    """Get text features from ViCLIP"""
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]

    text_template = f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_feature_dict[input_text] = text_features

    return text_features


def get_vid_features(model, input_frames):
    """Get video features from ViCLIP"""
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames, test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
    return clip_feat


def evaluate_semantic_on_portion(
    model,
    tokenizer,
    video_path,
    prompt,
    height=384,
    width=640,
    device="cuda",
    start_ratio=0.0,
    end_ratio=DRIFT_RATIO,
    num_frames=8,
):
    """Evaluate semantic consistency on a portion of the video"""
    image_transform = clip_transform(224)

    with torch.no_grad():
        # Load video frames from the specified portion
        images = load_video_frames(
            video_path, start_ratio, end_ratio, num_frames=num_frames, height=height, width=width
        )
        images = image_transform(images)
        images = images.to(device)

        # Get features
        clip_feat = get_vid_features(model, images.unsqueeze(0))
        text_feat = get_text_features(model, prompt, tokenizer)

        # Calculate similarity
        logit_per_text = clip_feat @ text_feat.T
        score = float(logit_per_text[0][0].cpu())

    return score


def evaluate_drifting_semantic(model, tokenizer, video_path, prompt, height=384, width=640, device="cuda"):
    """
    Evaluate drifting semantic consistency for a single video.
    Returns: (drift_score, start_score, end_score)
    """
    # Evaluate start portion (first 15%)
    start_score = evaluate_semantic_on_portion(
        model,
        tokenizer,
        video_path,
        prompt,
        height=height,
        width=width,
        device=device,
        start_ratio=0.0,
        end_ratio=DRIFT_RATIO,
        num_frames=8,
    )

    # Evaluate end portion (last 15%)
    end_score = evaluate_semantic_on_portion(
        model,
        tokenizer,
        video_path,
        prompt,
        height=height,
        width=width,
        device=device,
        start_ratio=1.0 - DRIFT_RATIO,
        end_ratio=1.0,
        num_frames=8,
    )

    # Calculate drift as absolute difference
    drift_score = abs(start_score - end_score)

    return drift_score, start_score, end_score


def main(args):
    baseline_name = os.path.basename(args.video_dir)
    output_path = os.path.join(args.output_path, baseline_name)
    output_json_path = os.path.join(output_path, "drifting_semantic_results.json")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CSV file
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    df_dict = df.set_index("id").to_dict("index")

    # Validate CSV columns
    required_columns = ["id", "duration", "prompt"]
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
            drift_scores.append(existing_results[video_id]["drift_semantic_score"])
        else:
            # Need to process
            prompt = df_dict[video_id]["prompt"]
            videos_to_process.append((video_path, video_id, video_name, prompt))

    print(f"Already processed: {len(existing_results)} videos")
    print(f"Need to process: {len(videos_to_process)} videos")

    # Process remaining videos
    if videos_to_process:
        # Load ViCLIP model
        print("Loading ViCLIP model...")
        tokenizer_path = os.path.join(args.semantic_model_path, "bpe_simple_vocab_16e6.txt.gz")
        semantic_model_path = os.path.join(args.semantic_model_path, "ViClip-InternVid-10M-FLT.pth")

        tokenizer = SimpleTokenizer(tokenizer_path)
        viclip = ViCLIP(tokenizer=tokenizer, pretrain=semantic_model_path).to(device)
        viclip.eval()

        print("\nEvaluating remaining videos...")
        for video_path, video_id, video_name, prompt in tqdm(videos_to_process):
            try:
                drift_score, start_score, end_score = evaluate_drifting_semantic(
                    viclip,
                    tokenizer,
                    video_path,
                    prompt,
                    height=args.height,
                    width=args.width,
                    device=device,
                )

                result_item = {
                    "id": video_id,
                    "video_name": video_name,
                    "prompt": prompt,
                    "drift_semantic_score": drift_score,
                    "start_semantic_score": start_score,
                    "end_semantic_score": end_score,
                }
                results.append(result_item)
                drift_scores.append(drift_score)

            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
                continue
    else:
        print("No videos to process. Skipping evaluation.")
        return

    # Sort all results by video_id
    results_sorted = sorted(results, key=lambda x: x["id"])

    # Calculate overall metrics
    if drift_scores:
        avg_drift = sum(drift_scores) / len(drift_scores)

        output = {
            "metric": "drifting_semantic",
            "description": f"Start-end contrast of semantic consistency (first/last {DRIFT_RATIO * 100:.0f}% frames)",
            "average_drift_score": avg_drift,
            "num_videos": len(drift_scores),
            "per_video_results": results_sorted,
        }

        # Save results
        os.makedirs(output_path, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n{'=' * 60}")
        print("Results Summary:")
        print(f"{'=' * 60}")
        print(f"Average Drifting Semantic Score: {avg_drift:.4f}")
        print("(Lower is better - indicates less semantic drift)")
        print(f"Number of videos evaluated: {len(drift_scores)}")
        print(f"Results saved to: {output_json_path}")
        print(f"{'=' * 60}\n")
    else:
        print("No videos were successfully evaluated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate drifting semantic consistency using ViCLIP model")

    # Input/Output arguments
    parser.add_argument("--height", type=str, default=384)
    parser.add_argument("--width", type=str, default=640)
    parser.add_argument("--input_csv", type=str, default="playground/gigaworld_t2v_prompts.csv")
    parser.add_argument("--video_dir", type=str, default="playground/toy-video")
    parser.add_argument("--output_path", type=str, default="playground/results")

    # Model arguments
    parser.add_argument("--semantic_model_path", type=str, default="checkpoints/ViCLIP")

    args = parser.parse_args()

    main(args)
