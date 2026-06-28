#!/usr/bin/env python3
"""
New batch video wall:
- 4 gt (3V) videos: use as-is
- 24 ep (single V) videos: group 3 horizontally → 8 3V videos
- Total 12 videos → 4×3 grid, 16:9, 5s, 10fps
"""
import os
import subprocess
import shutil

SRC_DIR = "/Users/zhanqianwu/Documents/工作/工作文档/Giga/Giga_world_1/github_page/video/video_Wall"
TMP_DIR = os.path.join(SRC_DIR, "_tmp_new")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

TARGET_DUR = 5.0
SPEED = 8.1 / TARGET_DUR  # 1.62x speedup

# === 1) Prepare 3V videos ===

# 1a) gt videos (already 3V, 2880x544)
gt_files = [
    "gt_ep00000.mp4", "gt_ep00006.mp4", "gt_ep00013.mp4", "gt_ep00067.mp4"
]
for i, f in enumerate(gt_files):
    src = os.path.join(SRC_DIR, f)
    out = os.path.join(TMP_DIR, f"gt_{i:02d}.mp4")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src,
        "-vf", f"setpts=PTS/{SPEED},fps=10",
        "-an", "-t", str(TARGET_DUR),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out
    ]
    subprocess.run(cmd, check=True)
    print(f"  gt_{i:02d}.mp4 (from {f})")

# 1b) Single V videos: group 3 horizontally
ep_files = [f"ep{i:05d}.mp4" for i in range(24)]
for g in range(8):
    group = [ep_files[g*3 + j] for j in range(3)]
    out = os.path.join(TMP_DIR, f"ep_group_{g:02d}.mp4")

    # Build filter: speed up each, then hstack
    filter_parts = []
    for j, f in enumerate(group):
        src = os.path.join(SRC_DIR, f)
        filter_parts.append(f"[{j}:v]setpts=PTS/{SPEED},fps=10[v{j}]")
    filter_parts.append(f"[v0][v1][v2]hstack=inputs=3[v]")
    filter_complex = ";".join(filter_parts)

    inputs = ["-i", os.path.join(SRC_DIR, f) for f in group]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(TARGET_DUR),
        out
    ]
    subprocess.run(cmd, check=True)
    print(f"  ep_group_{g:02d}.mp4 (from {group})")

# === 2) Build 4×3 video wall ===
COLS, ROWS = 4, 3
# 16:9: 4W/3H = 16/9 → W/H = 12/9 = 4/3
# Total 1920×1080, cell 480×360
CW, CH = 480, 360
print(f"\nCell: {CW}x{CH} (4:3), Total: {COLS*CW}x{ROWS*CH} (16:9)")

all_files = sorted(os.listdir(TMP_DIR))
inputs_args = []
filter_parts = []
for i, f in enumerate(all_files):
    inputs_args.extend(["-i", os.path.join(TMP_DIR, f)])
    # Source is 2880×544 (5.29:1), cell is 480×360 (4:3)
    # Scale to fit height, crop center to fill width
    filter_parts.append(f"[{i}:v]scale=-1:{CH}:force_original_aspect_ratio=decrease,crop={CW}:{CH},fps=10[v{i}]")

positions = []
for r in range(ROWS):
    for c in range(COLS):
        x = c * CW
        y = r * CH
        positions.append(f"{x}_{y}")

layout = "|".join(positions)
filter_complex = ";".join(filter_parts) + f";{''.join(f'[v{i}]' for i in range(12))}xstack=inputs=12:layout={layout}[v]"

out_mp4 = os.path.join(SRC_DIR, "video_wall_new.mp4")
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
out_gif = os.path.join(SRC_DIR, "video_wall_new.gif")
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
print("\nDone.")