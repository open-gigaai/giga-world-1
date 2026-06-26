#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

SAM2_MODEL_DIR = "/shared_disk/models/huggingface/models--facebook--sam2-hiera-large"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_sam2(device: str):
    device_id = 0 if device.startswith("cuda") else -1
    return pipeline(
        "mask-generation",
        model=SAM2_MODEL_DIR,
        device=device_id,
        local_files_only=True,
    )


def infer_masks(generator, image: Image.Image, points_per_batch: int):
    return generator(image.convert("RGB"), points_per_batch=points_per_batch)


def overlay_masks(image: Image.Image, masks, alpha: float = 0.45):
    image_np = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = image_np.copy()

    if masks is None or len(masks) == 0:
        return image.convert("RGB")

    rng = np.random.default_rng(0)
    for mask in masks:
        mask_np = np.asarray(mask).astype(bool)
        color = rng.integers(0, 255, size=3, dtype=np.uint8).astype(np.float32)
        overlay[mask_np] = overlay[mask_np] * (1 - alpha) + color * alpha

    return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))


def collect_inputs(input_path: str):
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in VIDEO_EXTS
    )


def process_image(generator, image_path: Path, output_dir: Path, points_per_batch: int):
    image = Image.open(image_path).convert("RGB")
    outputs = infer_masks(generator, image, points_per_batch)
    masks = outputs.get("masks", [])
    overlay = overlay_masks(image, masks)

    output_path = output_dir / f"{image_path.stem}_sam2_auto.png"
    overlay.save(output_path)
    print(f"saved: {output_path}, masks: {len(masks)}")


def process_video(generator, video_path: Path, output_dir: Path, points_per_batch: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_dir / f"{video_path.stem}_sam2_auto.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        outputs = infer_masks(generator, image, points_per_batch)
        overlay = overlay_masks(image, outputs.get("masks", []))
        writer.write(cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"saved: {output_path}, frames: {frame_idx}")


def main():
    parser = argparse.ArgumentParser(description="SAM2 automatic mask generation for images/videos")
    parser.add_argument("--input", required=True, help="Input image/video path or directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = load_sam2(args.device)
    input_paths = collect_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No images or videos found in {args.input}")

    for input_path in input_paths:
        suffix = input_path.suffix.lower()
        if suffix in IMAGE_EXTS:
            process_image(generator, input_path, output_dir, args.points_per_batch)
        elif suffix in VIDEO_EXTS:
            process_video(generator, input_path, output_dir, args.points_per_batch)


if __name__ == "__main__":
    main()

# python /mnt/pfs/users/zhanqian.wu/code/Helios/tools/offload_data/sam2_test.py \
#   --input /mnt/pfs/users/zhanqian.wu/output/infer_like_train/functrl_like_train_001_1780042561_gen.mp4 \
#   --output_dir /mnt/pfs/users/zhanqian.wu/code/Helios/tools/offload_data/sam2_outputs