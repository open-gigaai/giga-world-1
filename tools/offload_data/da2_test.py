#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

HUGGINGFACE_MODEL_CACHE = "/shared_disk/models/huggingface"


MODEL_KEY_MAP = {
    "small": os.path.join(HUGGINGFACE_MODEL_CACHE, "models--Depth-Anything-V2-Small-hf"),
    "base": os.path.join(HUGGINGFACE_MODEL_CACHE, "models--Depth-Anything-V2-Base-hf"),
    "large": os.path.join(HUGGINGFACE_MODEL_CACHE, "models--Depth-Anything-V2-Large-hf"),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_da2(model_size: str, device: str):
    model_dir = MODEL_KEY_MAP[model_size]

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
    return model, image_processor


@torch.inference_mode()
def infer_depth(model, image_processor, image: Image.Image, device: str):
    image = image.convert("RGB")
    inputs = image_processor(
        images=[image],
        return_tensors="pt",
        do_rescale=True,
    )
    inputs = {
        k: v.to(device, dtype=model.dtype, non_blocking=True)
        for k, v in inputs.items()
    }

    outputs = model(**inputs)
    depth = outputs.predicted_depth.unsqueeze(1)
    depth = F.interpolate(
        depth,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    d_min = depth.amin(dim=(1, 2), keepdim=True)
    d_max = depth.amax(dim=(1, 2), keepdim=True)
    depth = (depth - d_min) / (d_max - d_min + 1e-6)
    depth = (depth * 255).clamp(0, 255).to(torch.uint8)[0]

    depth_np = depth.cpu().numpy()
    return Image.fromarray(np.stack([depth_np] * 3, axis=-1))


def collect_inputs(input_path: str):
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in VIDEO_EXTS
    )


def process_image(model, image_processor, image_path: Path, output_dir: Path, device: str):
    image = Image.open(image_path)
    depth_image = infer_depth(model, image_processor, image, device)
    output_path = output_dir / f"{image_path.stem}_depth.png"
    depth_image.save(output_path)
    print(f"saved: {output_path}")


def process_video(model, image_processor, video_path: Path, output_dir: Path, device: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    duration = max(1, int(1000 / fps))
    output_path = output_dir / f"{video_path.stem}_depth.gif"

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_image = infer_depth(model, image_processor, image, device)
        frames.append(depth_image)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"saved: {output_path}, frames: {len(frames)}")


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 image/video inference script")
    parser.add_argument("--input", required=True, help="Input image/video path or directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_size", choices=["small", "base", "large"], default="small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, image_processor = load_da2(args.model_size, args.device)
    input_paths = collect_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No images or videos found in {args.input}")

    for input_path in input_paths:
        suffix = input_path.suffix.lower()
        if suffix in IMAGE_EXTS:
            process_image(model, image_processor, input_path, output_dir, args.device)
        elif suffix in VIDEO_EXTS:
            process_video(model, image_processor, input_path, output_dir, args.device)


if __name__ == "__main__":
    main()

# python /mnt/pfs/users/zhanqian.wu/code/Helios/tools/offload_data/da2_test.py \
#   --input /mnt/pfs/users/zhanqian.wu/output/infer_like_train/functrl_like_train_001_1780042561_gen.mp4 \
#   --output_dir /mnt/pfs/users/zhanqian.wu/code/Helios/tools/offload_data/da2_outputs \
#   --model_size small
