#!/usr/bin/env python3
"""
18-cell wall (6x3), 16:9, 5s.
Each cell is 8:9 (close to square) — video content scaled to fit height, cropped to width.
Sources (all speed-up to 5s):
- 14 old episode_*.mp4 (4:1, 12.1s → 5s)
- 4 gt_ep*.mp4 (3V, 8.1s → 5s)
Total 18, drop the last 8 single ep (use only first 16 single V? No, we have 14+4=18).
Use all 14 + 4 = 18. Drop nothing.
"""
import os
import subprocess
import glob
import shutil

SRC_DIR = "/Users/zhanqianwu/Documents/工作/工作文档/Giga/Giga_world_1/github_page/video/video_Wall"
TMP_DIR = os.path.join(SRC_DIR, "_tmp_18")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

TARGET_DUR = 5.0

# Old episodes: 14 files, use first 14 (we have exactly 14, drop none)
OLD_FILES = sorted(glob.glob(os.path.join(SRC_DIR, "episode_*.mp4")))
OLD_FILES = [f for f in OLD_FILES if "copy" not in os.path.basename(f)]
print(f"Old episodes: {len(OLD_FILES)}")

# gt: 4 files (we have exactly 4)
GT_FILES = sorted(glob.glob(os.path.join(SRC_DIR, "gt_ep*.mp4")))
print(f"GT: {len(GT_FILES)}")

# === Speed up old episodes ===
print("\nSpeed up old episodes (14)...")
for f in OLD_FILES:
    name = os.path.basename(f).replace(".mp4", "_sp.mp4")
    out = os.path.join(TMP_DIR, name)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", f,
        "-vf", "setpts=PTS/2.42,fps=10",
        "-an", "-t", str(TARGET_DUR),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out
    ]
    subprocess.run(cmd, check=True)

# === Speed up gt (3V) ===
print("\nSpeed up gt (4)...")
for f in GT_FILES:
    name = "gt_" + os.path.basename(f).replace(".mp4", "_sp.mp4")
    out = os.path.join(TMP_DIR, name)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", f,
        "-vf", "setpts=PTS/1.62,fps=10",
        "-an", "-t", str(TARGET_DUR),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out
    ]
    subprocess.run(cmd, check=True)

# === Build 6×3 wall ===
COLS, ROWS = 6, 3
# 16:9: 6W/3H = 16/9 → W/H = 8/9 → CW:CH = 8:9
CW, CH = 480, 540
print(f"\nCell: {CW}x{CH} (8:9), Total: {COLS*CW}x{ROWS*CH} (16:9)")

# Order: 4 gt first, then 14 old episodes
all_files = sorted(os.listdir(TMP_DIR))
print(f"Total cells: {len(all_files)}")
assert len(all_files) == 18, f"Expected 18, got {len(all_files)}"

inputs_args = []
filter_parts = []
for i, f in enumerate(all_files):
    inputs_args.extend(["-i", os.path.join(TMP_DIR, f)])
    # Source ratio: gt is 2880×544 (~5.3:1), old is 1920×480 (4:1).
    # Cell is 480×540 (8:9). Scale to fit height, crop to fill width.
    filter_parts.append(
        f"[{i}:v]scale=-1:{CH}:force_original_aspect_ratio=decrease,crop={CW}:{CH},fps=10[v{i}]"
    )

positions = []
for r in range(ROWS):
    for c in range(COLS):
        positions.append(f"{c*CW}_{r*CH}")
layout = "|".join(positions)
filter_complex = ";".join(filter_parts) + f";{''.join(f'[v{i}]' for i in range(len(all_files)))}xstack=inputs={len(all_files)}:layout={layout}[v]"

out_mp4 = os.path.join(SRC_DIR, "video_wall.mp4")
cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
    *inputs_args,
    "-filter_complex", filter_complex,
    "-map", "[v]", "-an",
    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "-t", str(TARGET_DUR),
    out_mp4
]
subprocess.run(cmd, check=True)
print(f"\n  -> {out_mp4}")

# GIF
out_gif = os.path.join(SRC_DIR, "video_wall.gif")
cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
    "-i", out_mp4,
    "-vf", "fps=10,scale=1920:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4",
    "-t", str(TARGET_DUR),
    out_gif
]
subprocess.run(cmd, check=True)
print(f"  -> {out_gif}")

shutil.rmtree(TMP_DIR)
print("\nDone.")