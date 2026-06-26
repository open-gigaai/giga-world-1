#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/mnt/pfs/users/zhanqian.wu/code/Gigaworld")
DEFAULT_TASKS = [
    "fold_the_shirt_easy",
    "microwave",
    "task1",
    "task2",
    "task3",
    "task4",
    "task5",
    "task11",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 FunCtrl overfit inference for one or more Gigaworld tasks."
    )
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Task names or 'all'.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Override checkpoint for a single task run.")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="/shared_disk/users/zhanqian.wu/output/experiment/gigaworld",
    )
    parser.add_argument("--checkpoint_step", type=int, default=2000)
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="ablation_stage_1_post_giga_functrl_lora_wan22_5b_overfit",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/shared_disk/users/hao.li/project/giga-world-2-main/DiffSynth-Studio_0611/eval_overfit",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "scripts/training/configs/stage_1_post_functrl_wan22_5b_overfit.yaml"),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/shared_disk/users/zhanqian.wu/model/Wan2.2-Fun-5B-Control-diffusers",
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default="/mnt/pfs/users/zhanqian.wu/ckpt/wan22-5b_stage-1-16gpus-21k",
    )
    parser.add_argument("--prompt_type", choices=["long", "short"], default="long")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids. When more than one id is provided, samples are sharded across GPUs.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_on_missing_checkpoint", action="store_true")
    return parser.parse_args()


def normalize_tasks(tasks):
    if len(tasks) == 1 and tasks[0] == "all":
        return DEFAULT_TASKS

    unknown = sorted(set(tasks) - set(DEFAULT_TASKS))
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Valid tasks: {DEFAULT_TASKS}")
    return tasks


def checkpoint_for_task(args, task):
    if args.checkpoint_path:
        return Path(args.checkpoint_path)

    checkpoint_dir = (
        Path(args.checkpoint_root)
        / f"{args.checkpoint_prefix}_{task}"
        / f"checkpoint-{args.checkpoint_step}"
    )
    return checkpoint_dir


def item_stem(item):
    raw_path = item.get("path") or item.get("control_path") or ""
    return Path(raw_path).stem


def select_caption_items(captions_path, prompt_type, max_samples):
    with open(captions_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    grouped = {}
    for item in raw_items:
        stem = item_stem(item)
        if not stem:
            continue
        grouped.setdefault(stem, {})[item.get("prompt_type", "long")] = item

    selected = []
    for stem in sorted(grouped):
        item = grouped[stem].get(prompt_type) or grouped[stem].get("long") or next(iter(grouped[stem].values()))
        prompt = (item.get("cap") or [""])[0]
        if not prompt:
            continue

        normalized = dict(item)
        normalized["cap"] = [prompt]
        normalized["path"] = f"{Path(item.get('path', '')).parent.name}/{stem}.mp4"
        normalized["control_path"] = f"{Path(item.get('control_path', '')).parent.name}/{stem}.mp4"
        selected.append(normalized)

        if max_samples is not None and len(selected) >= max_samples:
            break

    return selected, grouped


def write_task_captions(output_root, task, selected_items):
    manifest_dir = Path(output_root) / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    task_captions_path = manifest_dir / f"{task}_captions.json"
    with open(task_captions_path, "w", encoding="utf-8") as f:
        json.dump(selected_items, f, ensure_ascii=False, indent=2)
    return task_captions_path


def write_task_shard_captions(output_root, task, shard_id, selected_items):
    manifest_dir = Path(output_root) / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    task_captions_path = manifest_dir / f"{task}_shard{shard_id:02d}_captions.json"
    with open(task_captions_path, "w", encoding="utf-8") as f:
        json.dump(selected_items, f, ensure_ascii=False, indent=2)
    return task_captions_path


def captions_by_type(grouped, stem):
    captions = {}
    for prompt_type, item in grouped.get(stem, {}).items():
        cap = (item.get("cap") or [""])[0]
        if cap:
            captions[prompt_type] = cap
    return captions


def build_manifest(args, tasks, task_items):
    items = []
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    for task in tasks:
        selected_items, grouped = task_items[task]
        for item in selected_items:
            stem = item_stem(item)
            output_video = output_root / task / "videos" / f"{stem}.mp4"
            selected_caption = (item.get("cap") or [""])[0]
            items.append(
                {
                    "id": f"{task}/{stem}",
                    "task": task,
                    "stem": stem,
                    "input_image": str(dataset_root / task / "videos" / f"{stem}.mp4"),
                    "control_video": str(dataset_root / task / "control_videos" / f"{stem}.mp4"),
                    "output_video": str(output_video),
                    "captions": captions_by_type(grouped, stem),
                    "selected_caption": selected_caption,
                    "checkpoint": str(checkpoint_for_task(args, task)),
                }
            )

    manifest = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "prompt_type": args.prompt_type,
        "checkpoint_step": args.checkpoint_step,
        "num_inference_steps": args.num_inference_steps,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "tasks": tasks,
        "items": items,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "captions_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def parse_gpu_ids(gpu_ids):
    if not gpu_ids:
        return []
    return [gpu.strip() for gpu in gpu_ids.split(",") if gpu.strip()]


def build_infer_cmd(args, task, task_captions_path):
    checkpoint_path = checkpoint_for_task(args, task)
    output_dir = Path(args.output_root) / task
    dataset_dir = Path(args.dataset_root) / task

    return [
        sys.executable,
        str(PROJECT_ROOT / "infer/infer_benchmark.py"),
        "--config",
        args.config,
        "--base_model_path",
        args.base_model_path,
        "--transformer_model_name_or_path",
        args.transformer_model_name_or_path,
        "--checkpoint_path",
        str(checkpoint_path),
        "--dataset_dir",
        str(dataset_dir),
        "--captions_path",
        str(task_captions_path),
        "--control_video_path",
        "None",
        "--image_path",
        "None",
        "--prompt",
        "None",
        "--output_dir",
        str(output_dir),
        "--sample_name",
        "wan22_overfit",
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--num_frames",
        str(args.num_frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--enable_tiling",
        "True",
    ]


def run_task(args, task, task_captions_path):
    checkpoint_path = checkpoint_for_task(args, task)
    if not checkpoint_path.exists():
        message = f"Missing checkpoint for {task}: {checkpoint_path}"
        if args.fail_on_missing_checkpoint:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}; skip this task.")
        return

    if args.skip_existing and all_expected_outputs_exist(args, task, task_captions_path):
        print(f"[INFO] Skip {task}: all outputs already exist.")
        return

    output_dir = Path(args.output_root) / task
    dataset_dir = Path(args.dataset_root) / task
    cmd = build_infer_cmd(args, task, task_captions_path)

    print("=" * 100)
    print(f"[RUN] task={task}")
    print(f"      checkpoint={checkpoint_path}")
    print(f"      dataset={dataset_dir}")
    print(f"      output={output_dir}")
    print("=" * 100)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def run_task_parallel(args, task, selected_items):
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids) <= 1:
        task_captions_path = write_task_captions(args.output_root, task, selected_items)
        return run_task(args, task, task_captions_path)

    checkpoint_path = checkpoint_for_task(args, task)
    if not checkpoint_path.exists():
        message = f"Missing checkpoint for {task}: {checkpoint_path}"
        if args.fail_on_missing_checkpoint:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}; skip this task.")
        return

    shards = [[] for _ in gpu_ids]
    for idx, item in enumerate(selected_items):
        shards[idx % len(gpu_ids)].append(item)

    procs = []
    for shard_id, (gpu_id, shard_items) in enumerate(zip(gpu_ids, shards)):
        if not shard_items:
            continue

        shard_captions_path = write_task_shard_captions(args.output_root, task, shard_id, shard_items)
        if args.skip_existing and all_expected_outputs_exist(args, task, shard_captions_path):
            print(f"[INFO] Skip {task} shard {shard_id} on GPU {gpu_id}: outputs already exist.")
            continue

        cmd = build_infer_cmd(args, task, shard_captions_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        print("=" * 100)
        print(f"[RUN] task={task} shard={shard_id} gpu={gpu_id} samples={len(shard_items)}")
        print(f"      checkpoint={checkpoint_path}")
        print(f"      captions={shard_captions_path}")
        print("=" * 100)
        procs.append((shard_id, gpu_id, subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)))

    failures = []
    for shard_id, gpu_id, proc in procs:
        returncode = proc.wait()
        if returncode != 0:
            failures.append((shard_id, gpu_id, returncode))

    if failures:
        raise RuntimeError(f"Failed shards for {task}: {failures}")


def all_expected_outputs_exist(args, task, task_captions_path):
    with open(task_captions_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return all((Path(args.output_root) / task / "videos" / f"{item_stem(item)}.mp4").exists() for item in items)


def main():
    args = parse_args()
    tasks = normalize_tasks(args.tasks)
    task_items = {}
    task_captions_paths = {}

    for task in tasks:
        captions_path = Path(args.dataset_root) / task / "captions.json"
        if not captions_path.exists():
            raise FileNotFoundError(f"Missing captions file: {captions_path}")
        selected_items, grouped = select_caption_items(captions_path, args.prompt_type, args.max_samples)
        task_items[task] = (selected_items, grouped)
        task_captions_paths[task] = write_task_captions(args.output_root, task, selected_items)

    manifest_path = build_manifest(args, tasks, task_items)
    print(f"[INFO] Wrote manifest: {manifest_path}")

    for task in tasks:
        selected_items, _ = task_items[task]
        run_task_parallel(args, task, selected_items)


if __name__ == "__main__":
    main()
