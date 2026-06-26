import argparse
import os
import subprocess
from pathlib import Path


def extract_first_n_frames(input_root, output_root, num_frames=81):
    input_path = Path(input_root)
    output_path = Path(output_root)

    for subfolder in input_path.iterdir():
        if not subfolder.is_dir():
            continue

        print(f"Processing folder: {subfolder.name}")

        output_subfolder = output_path / subfolder.name
        output_subfolder.mkdir(parents=True, exist_ok=True)

        video_files = list(subfolder.glob("*.mp4"))

        for i, video_file in enumerate(video_files, 1):
            original_name = video_file.stem
            if "_ori" in original_name:
                new_name = original_name.rsplit("_ori", 1)[0] + f"_ori{num_frames}.mp4"
            else:
                new_name = video_file.name

            output_file = output_subfolder / new_name
            if os.path.exists(output_file):
                print(f"Skipping existing file: {output_file}")
                continue

            cmd = [
                "ffmpeg",
                "-i",
                str(video_file),
                "-vframes",
                str(num_frames),  # Extract only first N frames
                "-map",
                "0",  # Copy all streams (video + audio)
                "-c",
                "copy",  # Try direct copy (fastest, preserves all parameters)
                "-y",
                str(output_file),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True)

                if result.returncode != 0:
                    print(f"    Direct copy failed for {video_file.name}, re-encoding...")
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(video_file),
                        "-vframes",
                        str(num_frames),
                        "-c:v",
                        "libx264",  # Re-encode video
                        "-qp",
                        "0",  # Lossless quality
                        "-c:a",
                        "copy",  # Copy audio directly
                        "-map",
                        "0",  # Copy all streams
                        "-y",
                        str(output_file),
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)

                print(f"  [{i}/{len(video_files)}] {video_file.name} -> {new_name}")
            except subprocess.CalledProcessError as e:
                print(f"  Error processing {video_file.name}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first N frames from videos")
    parser.add_argument(
        "--input",
        type=str,
        default="long",
        help="Input root directory (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="short/0_from_long",
        help="Output root directory (default: ./output_frames)",
    )
    parser.add_argument("--frames", type=int, default=81, help="Number of frames to extract (default: 81)")

    args = parser.parse_args()

    extract_first_n_frames(args.input, args.output, args.frames)
    print("\nDone!")
