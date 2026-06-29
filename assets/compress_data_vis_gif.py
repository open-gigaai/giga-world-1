#!/usr/bin/env python3
"""Compress assets/data_vis.gif for GitHub (< 50 MB recommended).

Strategy:
  1. Try Pillow (PIL) — re-save the GIF with an adaptive palette, smaller
     width, and a configurable number of colors.
  2. Try imageio + imageio-ffmpeg — convert to WebM / MP4 (much smaller).
  3. Print the final file size and quality stats.

Usage:
    python assets/compress_data_vis_gif.py
    python assets/compress_data_vis_gif.py --target-mb 25
    python assets/compress_data_vis_gif.py --to webp
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "assets" / "data_vis.gif"
DST = REPO_ROOT / "assets" / "data_vis.gif.bak"  # backup of the original
TMP = REPO_ROOT / "assets" / "_data_vis.tmp.gif"


def _human_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.2f} MB"


def _print_status(label: str, before: int, after: int) -> None:
    delta = (1 - after / max(before, 1)) * 100
    print(
        f"  {label:<14} { _human_mb(before) } -> { _human_mb(after) }"
        f"   ({delta:+.1f}%)"
    )


def _backup() -> None:
    if not DST.exists():
        shutil.copy2(SRC, DST)
        print(f"[ok] backed up original to {DST.relative_to(REPO_ROOT)}")


def compress_with_pillow(target_mb: float, max_width: int) -> int:
    """Re-save the GIF with reduced palette, smaller width, and dispose=2."""
    from PIL import Image

    print("[pillow] reading source GIF...")
    img = Image.open(SRC)
    n_frames = getattr(img, "n_frames", 1)
    print(f"[pillow] {n_frames} frames, size={img.size}, mode={img.mode}")

    # Optional down-scale to max_width while keeping aspect ratio.
    if img.size[0] > max_width:
        ratio = max_width / img.size[0]
        new_size = (max_width, max(round(img.size[1] * ratio), 1))
        print(f"[pillow] down-scaling to {new_size}")
    else:
        new_size = img.size

    # Re-quantize each frame to 128 colors (256 looks nicer but balloons size).
    colors = 128
    print(f"[pillow] re-quantizing to {colors} colors...")
    frames = []
    durations = []
    for i in range(n_frames):
        img.seek(i)
        frame = img.convert("RGB").resize(new_size, Image.LANCZOS)
        frame = frame.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
        frames.append(frame)
        durations.append(img.info.get("duration", 50))

    before = SRC.stat().st_size
    frames[0].save(
        TMP,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=img.info.get("loop", 0),
        optimize=True,
        disposal=2,  # restore-to-background → fewer ghost frames
    )
    after = TMP.stat().st_size
    _print_status("pillow", before, after)
    return after


def convert_to_video(format_: str) -> Path:
    """Convert GIF → MP4 / WebM using imageio-ffmpeg."""
    import imageio
    import imageio_ffmpeg  # type: ignore

    reader = imageio.get_reader(SRC)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 20)
    if fps <= 0:
        fps = 20

    out_path = SRC.with_suffix("." + format_)
    if format_ == "mp4":
        writer = imageio_ffmpeg.write_frames(
            str(out_path), (out_path, "mp4"), fps=fps, quality=8
        )
        writer.send(None)
    elif format_ == "webm":
        writer = imageio_ffmpeg.write_frames(
            str(out_path), (out_path, "vp9"), fps=fps, quality=8
        )
        writer.send(None)
    else:
        raise ValueError(f"unsupported video format: {format_}")

    before = SRC.stat().st_size
    for frame in reader:
        writer.send(frame)
    writer.close()
    after = out_path.stat().st_size
    _print_status(f"video.{format_}", before, after)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-mb", type=float, default=25.0,
                        help="Stop compressing once smaller than this size")
    parser.add_argument("--max-width", type=int, default=960,
                        help="Maximum output width in pixels")
    parser.add_argument("--to", choices=("gif", "webp", "mp4", "webm"),
                        default="gif", help="Output format")
    args = parser.parse_args()

    if not SRC.exists():
        print(f"[err] {SRC} not found", file=sys.stderr)
        return 1

    target_bytes = int(args.target_mb * 1024 * 1024)
    print(f"[info] source: {SRC} ({_human_mb(SRC.stat().st_size)})")
    print(f"[info] target: < {args.target_mb} MB, max-width={args.max_width}, to={args.to}")
    _backup()

    if args.to in ("gif", "webp"):
        size = compress_with_pillow(args.target_mb, args.max_width)
        # Replace original if smaller.
        if size < SRC.stat().st_size:
            shutil.move(TMP, SRC)
            print(f"[ok] {SRC.relative_to(REPO_ROOT)} -> {_human_mb(size)}")
            if size > target_bytes:
                print(f"[warn] still larger than {args.target_mb} MB; "
                      "consider --max-width 720 or --to webp")
        else:
            TMP.unlink(missing_ok=True)
            print("[warn] compression did not help, original kept")
    else:
        out = convert_to_video(args.to)
        print(f"[ok] wrote {out.relative_to(REPO_ROOT)} ({_human_mb(out.stat().st_size)})")
        if out.stat().st_size > target_bytes:
            print(f"[warn] still larger than {args.target_mb} MB; "
                  "consider lowering --max-width")

    return 0


if __name__ == "__main__":
    sys.exit(main())
