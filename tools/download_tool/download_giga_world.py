#!/usr/bin/env python3
"""One-click downloader for Giga-World-1 models and toy data."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


HF_REPOS = {
    "model": "GigaAI-Research/Giga-World-1",
    "toydata": "GigaAI-Research/Giga-World-1-Toydata",
}

MODELSCOPE_REPOS = {
    "model": "GigaAI/Giga-World-1",
    "toydata": "GigaAI/Giga-World-1-Toydata",
}


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_from_hf(target: str, output_dir: Path, token: str | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with `pip install huggingface_hub`."
        ) from exc

    repo_id = HF_REPOS[target]
    local_dir = output_dir / repo_id.split("/")[-1]
    print(f"Downloading Hugging Face {target}: {repo_id}")
    print(f"Output: {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset" if target == "toydata" else "model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token or os.environ.get("HF_TOKEN"),
        resume_download=True,
    )


def download_from_modelscope(target: str, output_dir: Path, token: str | None = None) -> None:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: modelscope. Install it with `pip install modelscope`."
        ) from exc

    if token:
        os.environ["MODELSCOPE_TOKEN"] = token

    repo_id = MODELSCOPE_REPOS[target]
    local_dir = output_dir / repo_id.split("/")[-1]
    print(f"Downloading ModelScope {target}: {repo_id}")
    print(f"Output: {local_dir}")

    snapshot_download(
        repo_id,
        repo_type="dataset" if target == "toydata" else "model",
        local_dir=str(local_dir),
    )


def download_with_git(platform: str, target: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if platform == "hf":
        repo_id = HF_REPOS[target]
        url = f"https://huggingface.co/{'datasets/' if target == 'toydata' else ''}{repo_id}"
    else:
        repo_id = MODELSCOPE_REPOS[target]
        url = f"https://modelscope.cn/{'datasets' if target == 'toydata' else 'models'}/{repo_id}.git"

    run(["git", "lfs", "install"])
    run(["git", "clone", url, str(output_dir / repo_id.split("/")[-1])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Giga-World-1 model or toy data from Hugging Face / ModelScope."
    )
    parser.add_argument(
        "--platform",
        choices=["hf", "modelscope"],
        required=True,
        help="Download platform: hf or modelscope.",
    )
    parser.add_argument(
        "--target",
        choices=["model", "toydata", "all"],
        required=True,
        help="Download target: model, toydata, or all.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save downloaded files.",
    )
    parser.add_argument(
        "--method",
        choices=["sdk", "git"],
        default="sdk",
        help="Download method. Default: sdk.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF_TOKEN or MODELSCOPE_TOKEN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = ["model", "toydata"] if args.target == "all" else [args.target]

    for target in targets:
        if args.method == "git":
            download_with_git(args.platform, target, output_dir)
        elif args.platform == "hf":
            download_from_hf(target, output_dir, args.token)
        else:
            download_from_modelscope(target, output_dir, args.token)

    print("Done.")


if __name__ == "__main__":
    main()
