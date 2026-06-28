#!/usr/bin/env python3
"""Stitch 14 videos into a 7x2 video wall (5s). Final aspect ratio 16:9."""
import os
import subprocess
import glob
import shutil

SRC_DIR = "/Users/zhanqianwu/Documents/工作/工作文档/Giga/Giga_world_1/github_page/video/video_Wall"
TMP_DIR = os.path.join(SRC_DIR, "_tmp_speed")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

# Source videos (exclude generated outputs)
files = sorted(glob.glob(os.path.join(SRC_DIR, "*.mp4")))
files = [f for f in files if "video_wall" not in os.path.basename(f)]
files = sorted(files)[:14]

# Speed up: original ~12.1s → 5s
SPEED = 12.1 / 5.0
TARGET_DUR = 5.0

# Layout: 7 cols × 2 rows, overall 16:9
# 7W / 2H = 16/9 → W/H = 32/63
# Use k=8: W=256, H=504 → total 1792×1008 (16:9)
COLS, ROWS = 7, 2
CW, CH = 256, 504

print(f"Cell: {CW}x{CH}, Total: {COLS*CW}x{ROWS*CH} (16:9)")

# 1) Speed up each video to 5s
print("Speeding up videos...")
for f in files:
    name = os.path.basename(f)
    out = os.path.join(TMP_DIR, name)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", f,
        "-vf", f"setpts=PTS/{SPEED},fps=10",
        "-an", "-t", str(TARGET_DUR),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out
    ]
    subprocess.run(cmd, check=True)

# 2) Build xstack filter — scale each source to fill cell (crop to fit)
inputs_args = []
filter_parts = []
for i, f in enumerate(sorted(os.listdir(TMP_DIR))):
    inputs_args.extend(["-i", os.path.join(TMP_DIR, f)])
    # Original is 4:1 (wide), cell is ~1:2 (tall) → scale to fit height, then crop center
    filter_parts.append(f"[{i}:v]scale=-1:{CH}:force_original_aspect_ratio=decrease,crop={CW}:{CH},fps=10[v{i}]")

# Positions: no padding, no border
positions = []
for r in range(ROWS):
    for c in range(COLS):
        x = c * CW
        y = r * CH
        positions.append(f"{x}_{y}")

layout = "|".join(positions)
filter_complex = ";".join(filter_parts) + f";{''.join(f'[v{i}]' for i in range(14))}xstack=inputs=14:layout={layout}[v]"

# 3) Build MP4
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
print(f"  -> {out_mp4}")

# 4) Build GIF at 1920x1080
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

# Cleanup
shutil.rmtree(TMP_DIR)
print("Done.")