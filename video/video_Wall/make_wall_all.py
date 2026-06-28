#!/usr/bin/env python3
"""
Build a 26-cell video wall (13x2) from:
- 14 old episode_*.mp4 (already 3V-like 4:1)
- 4 gt_ep*.mp4 (3V, 2880x544)
- 24 ep*.mp4 (single V, 960x544) → 8 groups of 3V via hstack
Total: 14 + 4 + 8 = 26 cells, 13 cols × 2 rows, 16:9, 5s, 10fps
"""
import os
import subprocess
import glob
import shutil

SRC_DIR = "/Users/zhanqianwu/Documents/工作/工作文档/Giga/Giga_world_1/github_page/video/video_Wall"
TMP_DIR = os.path.join(SRC_DIR, "_tmp_all")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

TARGET_DUR = 5.0

# Group 1: 14 old episodes (1920x480, 4:1) - duration 12.1s, speed = 12.1/5 = 2.42
OLD_FILES = sorted(glob.glob(os.path.join(SRC_DIR, "episode_*.mp4")))
OLD_FILES = [f for f in OLD_FILES if "copy" not in os.path.basename(f)]
print(f"Old episodes: {len(OLD_FILES)}")

# Group 2: 4 gt_ep (3V, 2880x544) - duration 8.1s, speed = 8.1/5 = 1.62
GT_FILES = sorted(glob.glob(os.path.join(SRC_DIR, "gt_ep*.mp4")))
print(f"GT (3V): {len(GT_FILES)}")

# Group 3: 24 ep (single V, 960x544) → 8 groups - duration 8.1s, speed = 1.62
EP_FILES = sorted(glob.glob(os.path.join(SRC_DIR, "ep*.mp4")))
EP_FILES = [f for f in EP_FILES if "gt_ep" not in os.path.basename(f)]
print(f"EP single: {len(EP_FILES)}")

# === Speed up old episodes ===
print("\nSpeed up old episodes...")
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
print("\nSpeed up gt...")
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

# === Build ep groups (3x hstack) ===
print("\nBuild ep 3V groups...")
for g in range(8):
    group = [EP_FILES[g*3 + j] for j in range(3)]
    out = os.path.join(TMP_DIR, f"ep_group_{g:02d}.mp4")
    filter_parts = []
    for j, f in enumerate(group):
        filter_parts.append(f"[{j}:v]setpts=PTS/1.62,fps=10[v{j}]")
    filter_parts.append("[v0][v1][v2]hstack=inputs=3[v]")
    filter_complex = ";".join(filter_parts)
    cmd_input = []
    for f in group:
        cmd_input.extend(["-i", f])
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *cmd_input,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(TARGET_DUR),
        out
    ]
    subprocess.run(cmd, check=True)

# === Build 13x2 wall ===
COLS, ROWS = 13, 2
# 16:9: 13W / 2H = 16/9 → W/H = 32/117 ≈ 0.274
# Use CW=480, CH=1754 → 6240×3508 (16:9, 13*2=26 cells, fit exactly)
# But CH too tall. Try 6240×3508 directly.
# Better: CW=240, CH=175 → 3120×350 (8.91:1 not 16:9)
# Need 16:9: if W=3120, H=1755 → too tall
# Use total 1920x1080, CW=1920/13≈147, CH=1080/2=540 → ~3.6:1 cell
# We need cells to fit 5:1 source. Use total 13x ratio:
# 13 cols, 2 rows. Each cell should fit source ~5:1 ratio.
# cell W:H = (16/9 * 2) / 13 = 32/117 ≈ 0.274 → very tall
# So fit source by width: cell W fixed, H = cellW * sourceH/sourceW * 2
# Source 2880x544 → ratio 5.29:1, total 16:9 → cellW = 1920/13=147.7, cellH = 27.9. Too short.
# Total 13:2 (rows:cols) with cells holding 5:1 source need 13*5:2*1 = 65:2 ≈ 32.5:1
# So 16:9 cannot hold 13x2 with 5:1 source cells without distortion.
# Solution: Don't enforce 16:9. Use the natural ratio.
# Natural: 13 cols * cellW = 13W, 2 rows * cellH = 2 * W/5 = 2W/5 → total = 13W : 0.4W = 32.5:1
# This is fine for a "wall". Use CW=400, CH=400/5=80 → 5200x160 (32.5:1)
# But user wanted 16:9 earlier. Conflicting.
# Solution: keep cells as 5:1 source, overall 32.5:1, ignore 16:9 (or set outer to 16:9 with letterbox).
# I'll keep 16:9 outer via letterbox approach: pad with black bars around.
# Actually simpler: fit source cells, output is whatever ratio.
# Going with: CW=480, CH=480/5*1=96 → total 6240x192 (32.5:1, no letterbox, clean).

CW = 480
CH = 96  # 5:1
print(f"\nCell: {CW}x{CH} (5:1), Total: {COLS*CW}x{ROWS*CH} ({COLS*CW/(ROWS*CH):.1f}:1)")

all_files = sorted(os.listdir(TMP_DIR))
print(f"Total cells: {len(all_files)}")
assert len(all_files) == 26, f"Expected 26, got {len(all_files)}"

inputs_args = []
filter_parts = []
for i, f in enumerate(all_files):
    inputs_args.extend(["-i", os.path.join(TMP_DIR, f)])
    filter_parts.append(f"[{i}:v]scale={CW}:{CH},fps=10[v{i}]")

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

# Cleanup
shutil.rmtree(TMP_DIR)
print("\nDone.")