#!/usr/bin/env python3
"""Read raw RGB frames from stdin, draw direction/time/frame overlay, write to stdout."""
import sys
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 480
FPS = 12
WINDOW = 23  # seconds

# Try to find a usable system font
FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
font_big = font_small = None
for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        try:
            font_big = ImageFont.truetype(fp, 42)
            font_small = ImageFont.truetype(fp, 22)
            break
        except Exception:
            continue
if font_big is None:
    font_big = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Bigger time + frame fonts, highlighted
_font_path = FONT_CANDIDATES[0] if os.path.exists(FONT_CANDIDATES[0]) else None
font_time = ImageFont.truetype(_font_path, 28) if _font_path else font_big
font_frame = ImageFont.truetype(_font_path, 28) if _font_path else font_big

FRAME_BYTES = W * H * 3
frame_idx = 0
buf = sys.stdin.buffer

def draw_forward_arrow(draw, cx, cy, color, size=46):
    """Right-pointing triangle (▶) with body line."""
    # Shaft
    draw.rectangle([cx - size, cy - 8, cx + size * 0.4, cy + 8], fill=color)
    # Triangle head
    head = [
        (cx + size * 0.3, cy - size * 0.7),
        (cx + size * 0.3, cy + size * 0.7),
        (cx + size, cy),
    ]
    draw.polygon(head, fill=color)

def draw_backward_arrow(draw, cx, cy, color, size=46):
    """Left-pointing triangle (◀) with body line."""
    draw.rectangle([cx - size * 0.4, cy - 8, cx + size, cy + 8], fill=color)
    head = [
        (cx - size * 0.3, cy - size * 0.7),
        (cx - size * 0.3, cy + size * 0.7),
        (cx - size, cy),
    ]
    draw.polygon(head, fill=color)

while True:
    raw = buf.read(FRAME_BYTES)
    if len(raw) < FRAME_BYTES:
        break
    img = Image.frombytes("RGB", (W, H), raw)
    draw = ImageDraw.Draw(img)

    # Background panel
    panel_x, panel_y, panel_w, panel_h = 10, 10, 560, 170
    draw.rectangle([panel_x, panel_y, panel_x + panel_w, panel_y + panel_h], fill=(0, 0, 0))

    t_sec = frame_idx / FPS

    # Direction: arrow + text
    if (t_sec % (2 * WINDOW)) < WINDOW:
        arrow_color = (51, 255, 119)
        label_text = "FORWARD"
        draw_forward_arrow(draw, 60, 50, arrow_color, size=44)
        draw.text((120, 25), label_text, font=font_big, fill=arrow_color)
    else:
        arrow_color = (255, 85, 119)
        label_text = "BACKWARD"
        draw_backward_arrow(draw, 60, 50, arrow_color, size=44)
        draw.text((120, 25), label_text, font=font_big, fill=arrow_color)

    # Time + Frame (same size 28, highlighted)
    hh = int(t_sec // 3600)
    mm = int((t_sec % 3600) // 60)
    ss = t_sec % 60
    time_str = f"TIME  {hh:02d}:{mm:02d}:{ss:05.2f}"
    frame_str = f"FRAME  {frame_idx:05d}"
    draw.text((28, 92), time_str, font=font_time, fill=(255, 255, 0))
    draw.text((28, 132), frame_str, font=font_frame, fill=(0, 255, 255))

    sys.stdout.buffer.write(img.tobytes())
    frame_idx += 1

sys.stderr.write(f"[annotate] processed {frame_idx} frames\n")
sys.stderr.flush()