#!/usr/bin/env python3
"""
16:9 GIF/video wall without black bars.
Rules:
- 3V videos: use as 3V.
- single-V videos: every 3 single-V videos are horizontally stitched into one 3V video.
- Final wall: 3 columns × 9 rows = 27 cells, 1920×1080, no black bars.
- Duration: 5s, speed-up instead of trimming.
"""
import os
import subprocess
import glob
import shutil
import re

SRC_DIR = "/Users/zhanqianwu/Documents/工作/工作文档/Giga/Giga_world_1/github_page/video/video_Wall"
TMP_DIR = os.path.join(SRC_DIR, "_tmp_16")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

TARGET_DUR = 5.0
FPS = 10
COLS, ROWS = 3, 9
CW, CH = 640, 120
CELL_COUNT = COLS * ROWS


def run(cmd):
    subprocess.run(cmd, check=True)


def duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ], text=True).strip()
    return float(out)

# Existing 3V videos.
gt_files = sorted(glob.glob(os.path.join(SRC_DIR, "gt_ep*.mp4")))
old_three_v_files = [
    f for f in sorted(glob.glob(os.path.join(SRC_DIR, "episode_*.mp4")))
    if "copy" not in os.path.basename(f)
]

# Single-V videos: only ep00000.mp4-style files, group every 3 horizontally.
single_v_files = sorted(glob.glob(os.path.join(SRC_DIR, "ep*.mp4")))
single_v_files = [
    f for f in single_v_files
    if re.fullmatch(r"ep\d{5}\.mp4", os.path.basename(f))
]

print(f"GT 3V videos: {len(gt_files)}")
print(f"old 3V-like videos: {len(old_three_v_files)}")
print(f"single-V videos: {len(single_v_files)} -> {len(single_v_files)//3} grouped 3V videos")

prepared = []

# Prefer GT and single-V groups, then fill remaining slots with old 3V-like videos.
# 1) GT 3V.
for i, src in enumerate(gt_files):
    dur = duration(src)
    speed = dur / TARGET_DUR
    out = os.path.join(TMP_DIR, f"a_gt_{i:02d}.mp4")
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src,
        "-vf", f"setpts=PTS/{speed},fps={FPS}",
        "-an", "-t", str(TARGET_DUR),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out
    ])
    prepared.append(out)

# 2) Single-V groups -> 3V.
for g in range(len(single_v_files) // 3):
    group = single_v_files[g * 3:g * 3 + 3]
    dur = duration(group[0])
    speed = dur / TARGET_DUR
    out = os.path.join(TMP_DIR, f"b_single_group_{g:02d}.mp4")
    cmd_input = []
    for f in group:
        cmd_input.extend(["-i", f])
    filter_complex = ";".join([
        f"[{j}:v]setpts=PTS/{speed},fps={FPS}[v{j}]" for j in range(3)
    ] + ["[v0][v1][v2]hstack=inputs=3[v]"])
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *cmd_input,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(TARGET_DUR),
        out
    ])
    prepared.append(out)

# 3) Fill remaining cells with old 3V-like videos.
for i, src in enumerate(old_three_v_files):
    if len(prepared) >= CELL_COUNT:
        break
    dur = duration(src)
    speed = dur / TARGET_DUR
    out = os.path.join(TMP_DIR, f"c_old_{i:02d}.mp4")
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src,
        "-vf", f"setpts=PTS/{speed},fps={FPS}",
        "-an", "-t", str(TARGET_DUR),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out
    ])
    prepared.append(out)

while len(prepared) < CELL_COUNT:
    prepared.append(prepared[len(prepared) % len(prepared)])
prepared = prepared[:CELL_COUNT]
print(f"Using cells: {len(prepared)} / {CELL_COUNT}")

# Build 3×9 wall, output 1920×1080. Use cover scaling to avoid black bars.
inputs_args = []
filter_parts = []
for i, src in enumerate(prepared):
    inputs_args.extend(["-i", src])
    filter_parts.append(
        f"[{i}:v]scale={CW}:{CH}:force_original_aspect_ratio=increase,"
        f"crop={CW}:{CH},fps={FPS}[v{i}]"
    )

positions = []
for r in range(ROWS):
    for c in range(COLS):
        positions.append(f"{c*CW}_{r*CH}")

filter_complex = ";".join(filter_parts) + ";" + "".join(f"[v{i}]" for i in range(CELL_COUNT)) + \
    f"xstack=inputs={CELL_COUNT}:layout={'|'.join(positions)}[v]"

out_mp4 = os.path.join(SRC_DIR, "video_wall.mp4")
run([
    "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
    *inputs_args,
    "-filter_complex", filter_complex,
    "-map", "[v]", "-an",
    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "-t", str(TARGET_DUR),
    out_mp4
])
print(f"MP4 -> {out_mp4}")

out_gif = os.path.join(SRC_DIR, "video_wall.gif")
run([
    "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
    "-i", out_mp4,
    "-vf", "fps=10,scale=1920:1080:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4",
    "-t", str(TARGET_DUR),
    out_gif
])
print(f"GIF -> {out_gif}")

shutil.rmtree(TMP_DIR)
print("Done.")