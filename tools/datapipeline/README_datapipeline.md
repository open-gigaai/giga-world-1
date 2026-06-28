# DataPipeline Guide

This directory contains the main LeRobot dataset processing script:

- `tools/datapipeline/datapipeline_lerobot.py`

The script converts raw LeRobot-style robot datasets into the training format used by this repository. It produces resized RGB videos, depth videos, and episode-level labels under `labels/data.pkl`.

## Path Convention

Data input and output paths are resolved relative to the project root by default. For example:

- `output`
- `origin_data/task1`
- `example/toy_train_dataset/nano`

Absolute paths are also supported.

Model checkpoints referenced inside the script, such as Qwen3-VL, are currently configured with absolute paths.

## What `datapipeline_lerobot.py` Does

For each task directory, the pipeline:

1. Reads episode metadata from `data/chunk-*/episode_*.parquet`
2. Reads RGB videos from three camera views under `videos/`
3. Loads short task descriptions from `meta/episodes.jsonl`
4. Generates dense long captions from the `cam_high` video using Qwen3-VL
5. Runs Depth Anything V2 on all three views to produce depth videos
6. Writes the final dataset as:
   - `gt/`
   - `depth/`
   - `labels/data.pkl`
   - `labels/config.json`
   - `config.json`

## Expected Input Layout

Each input task directory should follow the LeRobot-style structure below:

```text
task_name/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_high/
│       ├── observation.images.cam_left_wrist/
│       └── observation.images.cam_right_wrist/
└── meta/
    └── episodes.jsonl
```

You may also pass a parent directory that contains multiple task folders. The script will automatically keep only subdirectories that contain both `data/` and `videos/`.

## Command Line Usage

Recommended usage from the project root:

```bash
python tools/datapipeline/datapipeline_lerobot.py \
  --output_base output \
  --data_dir_list origin_data/task1 \
  --num_gpus 8 \
  --max_tasks -1
```

You can also run it from inside `tools/datapipeline/` because paths are still resolved relative to the project root:

```bash
cd tools/datapipeline
python datapipeline_lerobot.py \
  --output_base output \
  --data_dir_list origin_data/task1 \
  --num_gpus 8
```

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--output_base` | `output` | Base output directory |
| `--data_dir_list` | `origin_data` | One or more input data directories |
| `--num_gpus` | `8` | Number of GPUs used for parallel processing |
| `--max_tasks` | `-1` | Limit the number of episodes for debugging; `-1` means no limit |

## Configurable Constants in the Script

Several top-level constants in `datapipeline_lerobot.py` are intended to be edited when needed:

| Variable | Default | Description |
| --- | --- | --- |
| `VIDEO_FPS` | `30` | Output video FPS |
| `TARGET_HEIGHT` | `480` | Output video height |
| `TARGET_WIDTH` | `640` | Output video width |
| `CAPTION_MODEL_PATH` | local Qwen3-VL path | Qwen3-VL checkpoint path |
| `CAPTION_MAX_PIXELS` | `360 * 420` | Video tokenization limit for captioning |
| `CAPTION_FPS` | `2.0` | Sampling FPS used when sending a clip to Qwen3-VL |
| `CAPTION_MAX_NEW_TOKENS` | `256` | Maximum caption generation length |
| `LONG_PROMPT_SEGMENT_FRAMES` | `300` | Number of frames per long-caption segment |

## Output Layout

A processed task is written as:

```text
output_dir/
├── gt/
│   ├── cam_high/
│   ├── cam_left_wrist/
│   └── cam_right_wrist/
├── depth/
│   ├── cam_high/
│   ├── cam_left_wrist/
│   └── cam_right_wrist/
├── labels/
│   ├── data.pkl
│   └── config.json
└── config.json
```

### Directory Semantics

| Path | Description |
| --- | --- |
| `gt/` | Resized RGB videos for the three camera views |
| `depth/` | Depth Anything V2 outputs, stored as videos with the same structure as `gt/` |
| `labels/data.pkl` | Episode-level metadata used by training and downstream tooling |
| `labels/config.json` | Per-task dataset config |
| `config.json` | Root-level dataset config |

## `labels/data.pkl` Format

`labels/data.pkl` is a Python `list`. Each element is a `dict` for one episode.

A typical record looks like:

```python
{
    "action": List[List[float]],
    "data_index": int,
    "episode_name": str,

    "cam_high_video_path": str,
    "cam_left_wrist_video_path": str,
    "cam_right_wrist_video_path": str,

    "cam_high_depth_path": str,
    "cam_left_wrist_depth_path": str,
    "cam_right_wrist_depth_path": str,

    "qpos": List[List[float]],
    "video_height": int,
    "video_width": int,
    "video_length": int,

    "short-prompt": {
        "task1": {
            "start_idx": "0",
            "end_idx": "299",
            "description": "put banana into basket"
        }
    },

    "long-prompt": {
        "long prompt 1": {
            "start_idx": "0",
            "end_idx": "299",
            "caption": "The robot arm reaches toward ..."
        }
    }
}
```

### Prompt Fields

- `short-prompt`
  - Loaded from `meta/episodes.jsonl`
  - Each `taskN` entry stores `start_idx`, `end_idx`, and `description`

- `long-prompt`
  - Generated by Qwen3-VL from the `cam_high` video
  - Split into multiple segments named `long prompt N`
  - Each segment stores `start_idx`, `end_idx`, and `caption`

## Config Files Written by the Pipeline

### `labels/config.json`

```json
{
  "_class_name": "PklDataset",
  "_key_names": ["action", "data_index", "episode_name", "..."],
  "data_size": 8
}
```

### `config.json`

```json
{
  "_class_name": "Dataset",
  "config_paths": ["labels/config.json"]
}
```

## Resume Behavior

The pipeline is resumable at both the task level and the episode level.

- If `labels/data.pkl` already exists for a task, that task is skipped entirely.
- If all RGB and depth videos already exist for a specific episode, video writing is skipped for that episode.
- Caption results are cached temporarily in `labels/caption_cache.json` during processing.
- The caption cache is deleted after final aggregation, so a future full rerun will regenerate it.

## Multi-GPU Processing Behavior

The script uses `torch.multiprocessing.spawn` and processes one task at a time.

For each task:

1. Episodes are collected into a task-local list.
2. Work is split across ranks using `task_batch[rank::world_size]`.
3. Each rank loads:
   - one Depth Anything V2 model
   - one Qwen3-VL model
4. Each rank writes a temporary `.rank_<rank>_results.pkl` file.
5. The main process aggregates all rank outputs and writes the final labels.
6. Temporary files are deleted immediately after aggregation.

This design keeps memory scoped to a single task and avoids holding processed metadata for multiple tasks at once.

## Direct Python Dependencies

`datapipeline_lerobot.py` directly imports:

```python
from thirdparty.depth_abything_v2.pipeline_da2 import (
    get_depth_image_batch_da2,
    get_image_depth_anything,
)
from tools.image_utils import load_video_frames, save_frames
```

It also loads the following at runtime:

```python
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info
```

## External Models

Make sure these model assets are available locally before running the pipeline.

| Model | Where it is configured | Default path |
| --- | --- | --- |
| Depth Anything V2 Small | `thirdparty/model_config.py` | `/shared_disk/models/huggingface/models--Depth-Anything-V2-Small-hf` |
| Qwen3-VL-8B-Instruct | `datapipeline_lerobot.py` -> `CAPTION_MODEL_PATH` | `/shared_disk/models/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/` |

## Related Tools in This Repository

The current repository layout includes these nearby tool directories:

```text
tools/
├── datapipeline/
│   ├── datapipeline_lerobot.py
│   └── README_datapipeline.md
├── data_vis_tools/
│   ├── app.py
│   ├── README.md
│   └── piper/
├── offload_data/
└── others/
```

Only `datapipeline_lerobot.py` is part of the core dataset conversion flow documented here.

`tools/data_vis_tools/` is a separate visualization utility for inspecting processed data and URDF playback. It is not required to run the pipeline itself.

## End-to-End Example

```bash
# Run the pipeline from the project root.
python tools/datapipeline/datapipeline_lerobot.py \
  --output_base output \
  --data_dir_list origin_data/task1 \
  --num_gpus 4
```

If you want a compact open-source example dataset layout, see the example directories already added under:

- `example/toy_datapipeline_dataset/`
- `example/toy_train_dataset/nano`
- `example/toy_train_dataset/pro`

## Notes

- Relative paths are resolved against the repository root, not the current shell working directory.
- The pipeline assumes CUDA is available.
- If no valid data directory is found, the script exits early.
- If no GPU is available, the script exits early.
- The output directory is created automatically.
