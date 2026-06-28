# DataPipeline 使用说明

本目录（`tools/datapipeline/`）包含 LeRobot **数据处理**主脚本及说明文档。

**路径约定：** 数据目录、输出目录等默认使用**相对于项目根目录**的路径（如 `output`、`origin_data/task1`）；模型权重路径保持绝对路径（如 `/shared_disk/models/huggingface/...`）。

## 项目目录结构

```
tools/
├── datapipeline/
│   ├── datapipeline_lerobot.py    # 【数据处理】主脚本
│   └── README_datapipeline.md     # 本文档
├── key_pts_utils.py               # 【可选工具】EE 关键点 sketch
├── plucker_pts_utils.py           # 【可选工具】Plücker 射线可视化
├── urdf_vis_tools/                # 【可选工具】URDF 3D 可视化
├── cam_calib_tools/               # urdf_vis_tools 标定页依赖，数据处理不需要
├── robo_tools.py                  # 可选工具共用：URDF 正运动学
├── cam_utils.py                   # 可选工具共用：相机投影与 overlay
├── image_utils.py                 # 数据处理共用：视频读写
└── piper/                         # 机械臂 URDF / mesh
```

## 整体流程

```
LeRobot 原始数据
      │
      ▼
【数据处理】datapipeline_lerobot.py
      │  视频转码 + 深度估计 + 文本标注
      ▼
labels/data.pkl + gt/ + depth/
      │
      ├──────────────────── 可选，按需运行 ────────────────────┐
      ▼                          ▼                              ▼
key_pts_utils.py          plucker_pts_utils.py            urdf_vis_tools/
（关键点 sketch）          （Plücker 射线视频）            （3D 可视化 / 标定调试）
                                                                  │
                                                                  └── cam_calib_tools/
                                                                      （标定页计算逻辑）
```

---

## 目录

- [1. 数据处理：datapipeline_lerobot.py](#1-数据处理datapipeline_lerobotpy)
- [2. 数据处理依赖模块](#2-数据处理依赖模块)
  - [2.1 thirdparty/depth_abything_v2](#21-thirdpartydepth_abything_v2)
  - [2.2 tools/image_utils.py](#22-toolsimage_utilspy)
- [3. 可选工具](#3-可选工具)
  - [3.1 tools/key_pts_utils.py](#31-toolskey_pts_utilspy)
  - [3.2 tools/plucker_pts_utils.py](#32-toolsplucker_pts_utilspy)
  - [3.3 tools/urdf_vis_tools](#33-toolsurdf_vis_tools)
  - [3.4 tools/cam_calib_tools](#34-toolscam_calib_tools)
- [4. 完整运行示例](#4-完整运行示例)
- [5. 数据集标准格式](#5-数据集标准格式)
  - [5.1 数据集文件格式](#51-数据集文件格式)
  - [5.2 元数据标签格式（labels/data.pkl）](#52-元数据标签格式labelsdatapkl)
  - [5.3 配置文件格式](#53-配置文件格式)
  - [5.4 可选工具扩展字段](#54-可选工具扩展字段)
- [6. Python 环境依赖（汇总）](#6-python-环境依赖汇总)
- [7. 模块依赖关系图](#7-模块依赖关系图)

---

## 1. 数据处理：datapipeline_lerobot.py

将 **LeRobot 格式** 的机器人数据集处理为训练用格式：统一分辨率视频、深度图视频、短/长文本 prompt，并生成 `labels/data.pkl`。

### 功能概览

| 步骤 | 说明 |
|------|------|
| 读取 episode | 从 `data/chunk-*/episode_*.parquet` 读取 `action`、`observation.state` |
| 读取视频 | 从 `videos/` 下三个相机视角读取原始 mp4 |
| 短 prompt | 从 `meta/episodes.jsonl` 读取任务描述 |
| 长 prompt | 用 **Qwen3-VL** 对 `cam_high` 视频分段生成 dense caption |
| 深度估计 | 用 **Depth Anything V2** 对三视角视频逐帧估计深度 |
| 写出结果 | `gt/`、`depth/` 视频 + `labels/data.pkl` |

### 命令行参数

```bash
# 方式一：在项目根目录运行（推荐）
python tools/datapipeline/datapipeline_lerobot.py \
  --output_base output \
  --data_dir_list origin_data/task1 \
  --num_gpus 8 \
  --max_tasks -1

# 方式二：在本目录运行（路径仍相对项目根目录解析）
cd tools/datapipeline
python datapipeline_lerobot.py \
  --output_base output \
  --data_dir_list origin_data/task1 \
  --num_gpus 8
```

> 数据与输出路径默认相对于**项目根目录**解析；也可传入绝对路径。模型路径（如 Qwen3-VL）仍使用绝对路径。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_base` | `output` | 输出根目录（相对项目根目录） |
| `--data_dir_list` | `origin_data` | 输入数据目录列表；支持传父目录，自动展开含 `data/` + `videos/` 的子目录 |
| `--num_gpus` | `8` | 并行 GPU 数量 |
| `--max_tasks` | `-1` | 限制 episode 数量（调试用），`-1` 表示不限制 |

### 输入数据格式（LeRobot）

每个 task 目录需包含：

```
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

### 脚本内可修改的配置（文件顶部）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TARGET_HEIGHT` / `TARGET_WIDTH` | `480` / `640` | 输出视频分辨率 |
| `CAPTION_MODEL_PATH` | Qwen3-VL-8B-Instruct 本地路径 | 长 prompt 标注模型 |
| `LONG_PROMPT_SEGMENT_FRAMES` | `300` | 每段 caption 的帧数 |

### 直接 import 的依赖

```python
from thirdparty.depth_abything_v2.pipeline_da2 import get_depth_image_batch_da2, get_image_depth_anything
from tools.image_utils import load_video_frames, save_frames
```

运行时还会动态加载：

```python
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info
```

### 断点续跑

- 若 `{output_base}/{task}/labels/data.pkl` 已存在，整个 task 跳过
- 若单个 episode 的 6 个 gt/depth 视频都已存在，该 episode 视频处理跳过（caption 可走缓存）
- caption 中间结果缓存在 `labels/caption_cache.json`，聚合完成后删除

---

## 2. 数据处理依赖模块

### 2.1 thirdparty/depth_abything_v2

**路径：** `thirdparty/depth_abything_v2/`

**核心文件：** `pipeline_da2.py`

在 `datapipeline_lerobot.py` 中用于对 RGB 帧批量估计深度图，并保存为 mp4 视频。

#### 主要接口

```python
from thirdparty.depth_abything_v2.pipeline_da2 import (
    get_image_depth_anything,      # 加载 Depth Anything V2 模型
    get_depth_image_batch_da2,     # 批量推理，返回 PIL 深度图列表
)
```

| 函数 | 作用 |
|------|------|
| `get_image_depth_anything(device, type="da2_small")` | 加载模型与 processor；`type` 可选 `da2_small` / `da2_base` / `da2_large` |
| `get_depth_image_batch_da2(model, processor, images, device)` | 输入 `List[PIL.Image]`，输出归一化到 0–255 的 3 通道深度 PIL 图 |

#### 内部依赖链

```
pipeline_da2.py
├── thirdparty/model_config.py     # 模型本地路径配置
├── tools/utils.py                 # resize_with_pad 等图像工具
└── tools/image_utils.py           # concat_images 等
```

#### 模型路径（model_config.py）

深度模型默认从本地 HuggingFace 缓存加载（`local_files_only=True`）：

| key | 默认路径 |
|-----|----------|
| `depth-anything-v2-hf-small` | `/shared_disk/models/huggingface/models--Depth-Anything-V2-Small-hf` |
| `depth-anything-v2-hf-base` | `.../models--Depth-Anything-V2-Base-hf` |
| `depth-anything-v2-hf-large` | `.../models--Depth-Anything-V2-Large-hf` |

> 运行前请确认对应模型目录已下载到 `HUGGINGFACE_MODEL_CACHE`（默认 `/shared_disk/models/huggingface`）。

---

### 2.2 tools/image_utils.py

**被 `datapipeline_lerobot.py` 直接使用：**

| 函数 | 作用 |
|------|------|
| `load_video_frames(video_path)` | 读取 mp4 全部帧，返回 `np.ndarray` 列表（RGB） |
| `save_frames(frames_dict, save_dir, episode_name, fps, ...)` | 将多相机帧字典保存为 mp4 |

`frames_dict` 的 key 在 pipeline 中为 `right_camera` / `left_camera` / `head_camera`，分别写入 `cam_right_wrist` / `cam_left_wrist` / `cam_high` 子目录。

---

## 3. 可选工具

以下工具在数据处理完成后**按需运行**，不参与核心 `gt/` / `depth/` / `labels/` 的生成。

### 3.1 tools/key_pts_utils.py

根据 `qpos` 做双臂 FK，将末端执行器（EE）关键点投影到 `cam_high` 画面，生成 sketch 示意视频。

#### 依赖

```python
from robo_tools import *    # URDFFK、split_dual_arm_qpos、make_T 等
from cam_utils import *     # build_head_camera_pose_and_K、project_points_to_image、draw_ee_overlay 等
```

| 模块 | 路径 | 作用 |
|------|------|------|
| `robo_tools.py` | `tools/robo_tools.py` | URDF 正运动学（`URDFFK`）、双臂 qpos 拆分 |
| `cam_utils.py` | `tools/cam_utils.py` | 相机内外参、3D→2D 投影、EE overlay 绘制 |
| `piper/piper.urdf` | `tools/piper/piper.urdf` | 默认机械臂 URDF |

#### 用法

修改脚本顶部 `DATA_FOLDERS`，指向数据处理输出目录：

```python
DATA_FOLDERS = [
    "output/task1",
]
```

```bash
python tools/key_pts_utils.py
```

#### 输出

```
output/task1/sketch/cam_high/
├── episode_000000.mp4          # 黑底 EE 关键点视频
└── episode_000000_debug.mp4    # 叠加在原图上的 debug 视频
```

并更新 `labels/data.pkl`，追加 `sketch_video_path`、`sketch_overlay_video_path`。

---

### 3.2 tools/plucker_pts_utils.py

读取 `labels/data.pkl`，对左右臂末端位姿计算 **Plücker 坐标**（射线 moment + direction），生成 4 路可视化视频。

#### 用法

```python
DATA_FOLDERS = ["output/task1"]
```

```bash
python tools/plucker_pts_utils.py
```

#### 输出

```
output/task1/plucker/
├── episode_000000_left_moment.mp4
├── episode_000000_left_direction.mp4
├── episode_000000_right_moment.mp4
└── episode_000000_right_direction.mp4
```

并更新 `labels/data.pkl`，追加 `left_plucker_*_video_path`、`right_plucker_*_video_path` 四个字段。

---

### 3.3 tools/urdf_vis_tools

独立 Web 服务，用于**可视化检查**处理结果，不修改 `data.pkl`。

| 页面 | 地址 | 说明 |
|------|------|------|
| URDF 3D Viewer | `http://<host>:8090/` | 加载 pkl + URDF，播放双臂 3D 动画，叠加相机画面 |
| Camera Calibration | `http://<host>:8090/calib` | 相机内外参标定、多帧 overlay、3D FK 可视化 |

```bash
cd tools/urdf_vis_tools
python app.py --host 0.0.0.0 --port 8090
```

更详细的 API 与界面说明见 `tools/urdf_vis_tools/README.md`。

---

### 3.4 tools/cam_calib_tools

**数据处理不需要此目录。** 它仅被 `urdf_vis_tools` 的标定页（`/calib`）动态加载，提供相机标定计算逻辑。

```
urdf_vis_tools/app.py  ──import──▶  cam_calib_tools/app.py
```

可独立启动 Gradio 版标定工具：

```bash
python tools/cam_calib_tools/app.py --port 7860
```

标定页参数（`cam_pos`、`cam_forward`、内参 K 等）与 `key_pts_utils.py` / `plucker_pts_utils.py` 中的相机/机器人参数可参考对齐。

---

## 4. 完整运行示例

```bash
# 在项目根目录执行

# 【必做】数据处理：LeRobot → gt/depth/labels
python tools/datapipeline/datapipeline_lerobot.py \
  --output_base output \
  --data_dir_list origin_data/task1 \
  --num_gpus 4

# 【可选】生成 EE sketch 视频（先改 DATA_FOLDERS）
python tools/key_pts_utils.py

# 【可选】生成 Plücker 射线视频（先改 DATA_FOLDERS）
python tools/plucker_pts_utils.py

# 【可选】启动 3D 可视化检查
cd tools/urdf_vis_tools && python app.py --port 8090
```

---

## 5. 数据集标准格式

以下为 `datapipeline_lerobot.py` 产出的数据格式。可选工具运行后会在 `data.pkl` 中追加额外字段。

### 5.1 数据集文件格式

```
output_dir/
├── gt/                    # 真实采集视频（ground truth），子目录名为相机名
│   ├── cam_high/          # episode_000000.mp4
│   ├── cam_left_wrist/
│   └── cam_right_wrist/
├── depth/                 # Depth Anything V2 生成的深度视频，子目录名为相机名
│   ├── cam_high/
│   ├── cam_left_wrist/
│   └── cam_right_wrist/
└── labels/                # 标签与配置文件
    ├── data.pkl           # 完整元数据（动作、状态、路径等）
    └── config.json        # 数据集加载配置
```

| 目录 | 说明 |
|------|------|
| `gt/` | 三视角 RGB 视频，每个 episode 一个 mp4 |
| `depth/` | 三视角深度视频，目录结构与 `gt/` 对应 |
| `labels/` | episode 级元数据与 DataLoader 配置 |

运行可选工具后，目录还会包含 `sketch/`、`plucker/` 等（见 [5.4](#54-可选工具扩展字段)）。

---

### 5.2 元数据标签格式（labels/data.pkl）

`data.pkl` 是一个 Python `list`，每一项是一个 `dict`，代表一条 episode 数据。

```python
{
    "action": List[List[float]],          # 动作序列 (T, D)
    "qpos": List[List[float]],            # 关节状态序列 (T, D)，来自 observation.state
    "data_index": int,                    # 全局唯一编号：0, 1, 2, ...
    "episode_name": str,                  # 如 "episode_000000"

    "cam_high_video_path": str,           # .../gt/cam_high/episode_000000.mp4
    "cam_left_wrist_video_path": str,
    "cam_right_wrist_video_path": str,
    "cam_high_depth_path": str,           # .../depth/cam_high/episode_000000.mp4
    "cam_left_wrist_depth_path": str,
    "cam_right_wrist_depth_path": str,

    "video_height": int,                  # 480
    "video_width": int,                   # 640
    "video_length": int,                  # 帧数 T，等于 len(qpos)

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
            "caption": "The robot arm reaches toward..."
        }
    }
}
```

**`short-prompt`：** 从 LeRobot `meta/episodes.jsonl` 读取 `tasks`；每个 `taskN` 含 `start_idx`、`end_idx`、`description`。

**`long-prompt`：** 由 Qwen3-VL 对 `cam_high` 视频分段标注；每段 `long prompt N` 含 `start_idx`、`end_idx`、`caption`。

---

### 5.3 配置文件格式

#### `labels/config.json`

```json
{
    "_class_name": "PklDataset",
    "_key_names": ["action", "data_index", "episode_name", "..."],
    "data_size": 8
}
```

#### `config.json`（output_dir 根目录）

```json
{
    "_class_name": "Dataset",
    "config_paths": ["labels/config.json"]
}
```

---

### 5.4 可选工具扩展字段

`key_pts_utils.py` / `plucker_pts_utils.py` 运行后追加：

| 字段 | 来源 | 说明 |
|------|------|------|
| `sketch_video_path` | key_pts_utils | 黑底 EE 关键点 sketch 视频 |
| `sketch_overlay_video_path` | key_pts_utils | 叠加在原图上的 debug 视频 |
| `left_plucker_moment_video_path` | plucker_pts_utils | 左臂 Plücker moment |
| `left_plucker_direction_video_path` | plucker_pts_utils | 左臂 Plücker direction |
| `right_plucker_moment_video_path` | plucker_pts_utils | 右臂 Plücker moment |
| `right_plucker_direction_video_path` | plucker_pts_utils | 右臂 Plücker direction |

---

## 6. Python 环境依赖（汇总）

| 包 | 用途 |
|----|------|
| `torch` | 深度模型、Plücker 计算、多 GPU |
| `transformers` | Depth Anything V2、Qwen3-VL |
| `qwen_vl_utils` | Qwen3-VL 视频输入处理 |
| `numpy`, `pandas` | 数据处理 |
| `opencv-python` (`cv2`) | 视频读写 |
| `av`, `Pillow`, `imageio` | 视频/图像 I/O |
| `tqdm` | 可选工具进度条 |

### 外部模型（需提前下载到本地）

| 模型 | 配置位置 | 默认路径 |
|------|----------|----------|
| Depth Anything V2 Small | `thirdparty/model_config.py` | `/shared_disk/models/huggingface/models--Depth-Anything-V2-Small-hf` |
| Qwen3-VL-8B-Instruct | `datapipeline_lerobot.py` 顶部 `CAPTION_MODEL_PATH` | `/shared_disk/models/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/` |

---

## 7. 模块依赖关系图

```
【数据处理】
tools/datapipeline/datapipeline_lerobot.py
├── thirdparty/depth_abything_v2/pipeline_da2.py
│   ├── thirdparty/model_config.py
│   ├── tools/utils.py
│   └── tools/image_utils.py
├── tools/image_utils.py
├── transformers (Qwen3-VL)
└── qwen_vl_utils

【可选工具】
tools/key_pts_utils.py ──┬── tools/robo_tools.py
tools/plucker_pts_utils.py ┤── tools/cam_utils.py
                           └── tools/piper/piper.urdf

tools/urdf_vis_tools/app.py
├── tools/piper/piper.urdf
└── tools/cam_calib_tools/app.py    # 仅标定页需要
```
