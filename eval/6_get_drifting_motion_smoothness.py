import argparse
import glob
import json
import os
import re

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.third_party.amt.utils.build_utils import build_from_cfg
from utils.third_party.amt.utils.utils import InputPadder, check_dim_and_resize, img2tensor, tensor2img
from utils.utils import align_dimension


# Percentage of frames to use for start/end portions
DRIFT_RATIO = 0.15


class FrameProcess:
    def __init__(self, height=384, width=640):
        self.height = height
        self.width = width

    def get_frames(self, video_path):
        """Extract frames from MP4 video"""
        frame_list = []
        video = cv2.VideoCapture(video_path)

        original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_aspect_ratio = original_width / original_height

        if self.width > self.height:
            target_width = self.width
            target_height = int(self.width / original_aspect_ratio)
        else:
            target_height = self.height
            target_width = int(self.height * original_aspect_ratio)

        target_height = align_dimension(target_height, 2)
        target_width = align_dimension(target_width, 2)

        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (target_width, target_height))
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != [], "No frames extracted from video"
        return frame_list

    def extract_frame(self, frame_list, start_from=0):
        extract = []
        for i in range(start_from, len(frame_list), 2):
            extract.append(frame_list[i])
        return extract


class MotionSmoothnessEvaluator:
    def __init__(self, config, ckpt, height=384, width=640, device="cuda"):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.height = height
        self.width = width
        self.initialization()
        self.load_model()

    def load_model(self):
        """Load AMT model"""
        cfg_path = self.config
        ckpt_path = self.ckpt
        network_cfg = OmegaConf.load(cfg_path).network
        network_name = network_cfg.name
        print(f"Loading [{network_name}] from [{ckpt_path}]...")

        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def initialization(self):
        """Initialize parameters based on device"""
        if self.device.type == "cuda":
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(self.device)
        self.fp = FrameProcess(height=self.height, width=self.width)

    def compute_motion_score_on_frames(self, frames):
        """Calculate motion smoothness score for a list of frames"""
        if len(frames) < 4:
            return 0.0

        frame_list = self.fp.extract_frame(frames, start_from=0)
        if len(frame_list) < 2:
            return 0.0

        # Convert to tensors
        inputs = [img2tensor(frame).to(self.device) for frame in frame_list]

        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        scale = (
            self.anchor_resolution
            / (h * w)
            * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        )
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16

        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        # Frame interpolation
        iters = int(self.niters)
        for i in range(iters):
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)["imgt_pred"]
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs

        # Calculate VFI score
        outputs = padder.unpad(*outputs)
        outputs = [tensor2img(out) for out in outputs]
        vfi_score = self.vfi_score(frames, outputs)
        norm = (255.0 - vfi_score) / 255.0

        return norm

    def vfi_score(self, ori_frames, interpolate_frames):
        """Calculate video frame interpolation quality score"""
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)

        scores = []
        min_len = min(len(ori), len(interpolate))
        for i in range(min_len):
            scores.append(self.get_diff(ori[i], interpolate[i]))

        if len(scores) == 0:
            return 0.0
        return np.mean(np.array(scores))

    def get_diff(self, img1, img2):
        """Calculate absolute difference between two images"""
        img = cv2.absdiff(img1, img2)
        return np.mean(img)

    def evaluate_drifting(self, video_path):
        """
        Evaluate drifting motion smoothness for a single video.
        Returns: (drift_score, start_score, end_score)
        """
        frames = self.fp.get_frames(video_path)
        total_frames = len(frames)
        num_drift_frames = max(4, int(total_frames * DRIFT_RATIO))  # Need at least 4 frames for AMT

        # Extract start and end portions
        start_frames = frames[:num_drift_frames]
        end_frames = frames[-num_drift_frames:]

        # Calculate scores for each portion
        start_score = self.compute_motion_score_on_frames(start_frames)
        end_score = self.compute_motion_score_on_frames(end_frames)

        # Calculate drift as absolute difference
        drift_score = abs(start_score - end_score)

        return drift_score, start_score, end_score


def main(args):
    baseline_name = os.path.basename(args.video_dir)
    output_path = os.path.join(args.output_path, baseline_name)
    output_json_path = os.path.join(output_path, "drifting_motion_smoothness_results.json")

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
            drift_scores.append(existing_results[video_id]["drift_motion_smoothness_score"])
        else:
            # Need to process
            videos_to_process.append((video_path, video_id, video_name))

    print(f"Already processed: {len(existing_results)} videos")
    print(f"Need to process: {len(videos_to_process)} videos")

    # Process remaining videos
    if videos_to_process:
        # Load model
        print("Loading AMT model...")
        evaluator = MotionSmoothnessEvaluator(
            args.config, args.smoothness_model_path, height=args.height, width=args.width, device=device
        )

        print("\nEvaluating remaining videos...")
        for video_path, video_id, video_name in tqdm(videos_to_process):
            try:
                drift_score, start_score, end_score = evaluator.evaluate_drifting(video_path)

                result_item = {
                    "id": video_id,
                    "video_name": video_name,
                    "drift_motion_smoothness_score": float(drift_score),
                    "start_motion_smoothness_score": float(start_score),
                    "end_motion_smoothness_score": float(end_score),
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
            "metric": "drifting_motion_smoothness",
            "description": f"Start-end contrast of motion smoothness (first/last {DRIFT_RATIO * 100:.0f}% frames)",
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
        print(f"Average Drifting Motion Smoothness Score: {avg_drift:.4f}")
        print("(Lower is better - indicates less motion quality drift)")
        print(f"Number of videos evaluated: {len(drift_scores)}")
        print(f"Results saved to: {output_json_path}")
        print(f"{'=' * 60}\n")
    else:
        print("No videos were successfully evaluated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate drifting motion smoothness using AMT model")

    # Input/Output arguments
    parser.add_argument("--height", type=str, default=384)
    parser.add_argument("--width", type=str, default=640)
    parser.add_argument("--input_csv", type=str, default="playground/gigaworld_t2v_prompts.csv")
    parser.add_argument("--video_dir", type=str, default="playground/toy-video")
    parser.add_argument("--output_path", type=str, default="playground/results")

    # Model arguments
    parser.add_argument("--config", type=str, default="checkpoints/AMT-S.yaml")
    parser.add_argument("--smoothness_model_path", type=str, default="checkpoints/amt_model/amt-s.pth")

    args = parser.parse_args()

    main(args)
