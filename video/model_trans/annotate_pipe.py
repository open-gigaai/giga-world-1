#!/usr/bin/env python3
"""Read raw RGB frames from stdin, draw direction/time/frame overlay, write to stdout.

Usage (via env vars or default):
  W=1280 H=480 FPS=12 python3 annotate_pipe.py
"""
import sys
import os
from PIL import Image, ImageDraw, ImageFont

W = int(os.environ.get("W", "1280"))
H = int(os.environ.get("H", "480"))
FPS = int(os.environ.get("FPS", "12"))
WINDOW = int(os.environ.get("WINDOW", "23"))  # direction switch window in seconds
SHOW_DIR = os.environ.get("SHOW_DIR", "1") == "1"  # arrow + label
SHOW_BG = os.environ.get("SHOW_BG", "1") == "1"   # black panel background

FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

def load_font(size):
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()

# Font sizes scale with video height
SCALE = H / 480.0
font_big = load_font(int(42 * SCALE))
font_time = load_font(int(28 * SCALE))
font_frame = load_font(int(28 * SCALE))

PANEL_W = int(560 * SCALE)
PANEL_H = int(170 * SCALE)
PANEL_X = int(10 * SCALE)
PANEL_Y = int(10 * SCALE)
ARROW_CX = int(60 * SCALE)
ARROW_CY = int(50 * SCALE)
ARROW_SIZE = int(44 * SCALE)
TEXT_X = int(120 * SCALE)
TEXT_Y = int(25 * SCALE)
INFO_X = int(28 * SCALE)
TIME_Y = int(92 * SCALE)
FRAME_Y = int(132 * SCALE)

def draw_forward_arrow(draw, cx, cy, color, size):
    half = max(2, int(8 * SCALE))
    shaft_w = int(size * 1.4)
    draw.rectangle([cx - size, cy - half, cx + size * 0.4, cy + half], fill=color)
    head = [
        (cx + size * 0.3, cy - size * 0.7),
        (cx + size * 0.3, cy + size * 0.7),
        (cx + size, cy),
    ]
    draw.polygon(head, fill=color)

def draw_backward_arrow(draw, cx, cy, color, size):
    half = max(2, int(8 * SCALE))
    shaft_w = int(size * 1.4)
    draw.rectangle([cx - size * 0.4, cy - half, cx + size, cy + half], fill=color)
    head = [
        (cx - size * 0.3, cy - size * 0.7),
        (cx - size * 0.3, cy + size * 0.7),
        (cx - size, cy),
    ]
    draw.polygon(head, fill=color)

FRAME_BYTES = W * H * 3
frame_idx = 0
buf = sys.stdin.buffer

while True:
    raw = buf.read(FRAME_BYTES)
    if len(raw) < FRAME_BYTES:
        break
    img = Image.frombytes("RGB", (W, H), raw)
    draw = ImageDraw.Draw(img)

    if SHOW_BG:
        draw.rectangle([PANEL_X, PANEL_Y, PANEL_X + PANEL_W, PANEL_Y + PANEL_H], fill=(0, 0, 0))

    t_sec = frame_idx / FPS

    if SHOW_DIR:
        if (t_sec % (2 * WINDOW)) < WINDOW:
            arrow_color = (51, 255, 119)
            label_text = "FORWARD"
            draw_forward_arrow(draw, ARROW_CX, ARROW_CY, arrow_color, ARROW_SIZE)
        else:
            arrow_color = (255, 85, 119)
            label_text = "BACKWARD"
            draw_backward_arrow(draw, ARROW_CX, ARROW_CY, arrow_color, ARROW_SIZE)
        draw.text((TEXT_X, TEXT_Y), label_text, font=font_big, fill=arrow_color)

    hh = int(t_sec // 3600)
    mm = int((t_sec % 3600) // 60)
    ss = t_sec % 60
    time_str = f"TIME  {hh:02d}:{mm:02d}:{ss:05.2f}"
    frame_str = f"FRAME  {frame_idx:05d}"
    draw.text((INFO_X, TIME_Y), time_str, font=font_time, fill=(255, 255, 0))
    draw.text((INFO_X, FRAME_Y), frame_str, font=font_frame, fill=(0, 255, 255))

    sys.stdout.buffer.write(img.tobytes())
    frame_idx += 1

sys.stderr.write(f"[annotate] processed {frame_idx} frames at {W}x{H} {FPS}fps\n")
sys.stderr.flush()