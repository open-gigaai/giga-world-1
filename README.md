<div align="center">

<img src="assets/main_page.png" alt="GigaWorld-1 Teaser" width="100%" />

# GigaWorld-1: A Roadmap to World Models for Robot Policy Evaluation

**Open-source training, inference, data processing, checkpoint conversion, and LoRA merge workflow.**

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2511.19861-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.19861)
[![Project Page](https://img.shields.io/badge/Project-Page-blueviolet.svg?style=for-the-badge&logo=google-chrome&logoColor=white)](https://github.com/Yvonne-OH/Giga-World-1-projectpage)
[![HuggingFace Model](https://img.shields.io/badge/🤗_Model-Giga--World--1-FFD21E.svg?style=for-the-badge)](https://huggingface.co/GigaAI-Research/Giga-World-1)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-Giga--World--1--Toydata-FFD21E.svg?style=for-the-badge)](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata)
[![ModelScope Model](https://img.shields.io/badge/ModelScope-Model-624AFF.svg?style=for-the-badge)](https://modelscope.cn/models/GigaAI/Giga-World-1/summary)
[![ModelScope Dataset](https://img.shields.io/badge/ModelScope-Dataset-624AFF.svg?style=for-the-badge)](https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata)
[![WMBench](https://img.shields.io/badge/📊_Benchmark-WMBench_Coming_Soon-orange.svg?style=for-the-badge)](#-wmbench-benchmark)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-7B68EE.svg?style=for-the-badge)](#-citation)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](#1-environment-setup)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](#1-environment-setup)
[![Diffusers](https://img.shields.io/badge/Diffusers-Custom-FFD21E.svg?style=flat-square)](#5-model-merge--checkpoint-conversion)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux-black.svg?style=flat-square&logo=linux&logoColor=white)](#1-environment-setup)

</div>

---

## 📰 Latest Updates

<div align="center">

| Date | Update |
| :---: | --- |
| 🆕 **2026-06-29** | 🎉 Initial open-source release of **Giga-World Release**: training (Stage-1 + Stage-2 DMD), inference (i2v / t2v), LoRA merge tool, and Web visualization tool. |
| 📦 **2026-06-28** | Project page live: [yvonne-oh.github.io/Giga-World-1-projectpage](https://yvonne-oh.github.io/Giga-World-1-projectpage). |
| 🏆 **2026-06** | 🎓 GigaWorld-1 paper accepted to **CVPR 2026**. See [Citation](#-citation) below. |
| 📄 **2025-11-25** | GigaWorld-0 (predecessor) released on arXiv: [arxiv.org/abs/2511.19861](https://arxiv.org/abs/2511.19861). |

</div>

> 💡 **Subscribe to releases** — click **Watch ▾ → Custom → Releases** on the GitHub repo to be notified when new weights, datasets, or the WMBench benchmark drop.

---

## 📊 Open-Source Progress

> 🟢 **Released** · 🟡 **Beta** · 🔴 **Coming Soon** — last updated 2026-06-29

| Status | Component | Description |
| :---: | --- | --- |
| 🟢 | **Training code — Stage-1** | `train_gigaworld_functrl_uni_stage1.py` for Nano (1.3B) and Pro (5B), DeepSpeed ZeRO-2/3 ready — see [§3.1](#31-stage-1-training-controllable-pre-training) |
| 🟢 | **Training code — Stage-2 DMD** | `train_gigaworld_functrl_uni_stage2_dmd.py` for DMD2 distillation (4–6 steps) — see [§3.2](#32-stage-2-dmd-training-acceleration-distillation) |
| 🟢 | **Inference (i2v / t2v)** | Nano + Pro one-click scripts, 10 FPS, 33 s rollouts — see [§6](#6--inference) |
| 🟢 | **Data preprocessing pipeline** | LeRobot-style → GigaWorld format with Qwen3-VL captions + Depth Anything V2 — see [§2.4](#24-lerobot-raw-data-preprocessing-pipeline) |
| 🟢 | **Toy training datasets** | `example/toy_train_dataset/nano` and `…/pro` for out-of-the-box training — see [§2.3](#23-use-the-bundled-toy-datasets) |
| 🟢 | **LoRA merge tool** | Unified `uni_merge_lora_for_giga_world_1.py` with HTML merge report — see [§5](#5--model-merge--checkpoint-conversion) |
| 🟢 | **URDF + Camera-Calibration Web tool** | Three.js dual-page viewer (`:8090/` and `:8090/calib`) — see [§4](#4-%EF%B8%8F-data--trajectory-visualization) |
| 🟢 | **Custom Triton / Flash kernels** | fp32_rmsnorm, triton_norm, triton_rope, tiled_linear |
| 🟢 | **🤗 Model weights on Hugging Face** | Nano (1.3B) and Pro (5B) Stage-1 checkpoints — see [GigaAI-Research/Giga-World-1](https://huggingface.co/GigaAI-Research/Giga-World-1) |
| 🟢 | **🤗 Toy data on Hugging Face** | Inference assets, raw LeRobot-format toy data, and toy training data — see [GigaAI-Research/Giga-World-1-Toydata](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata) |
| 🟢 | **ModelScope mirrors** | Model and toy data mirrors are available on [ModelScope Model](https://modelscope.cn/models/GigaAI/Giga-World-1/summary) and [ModelScope Dataset](https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata) |
| 🟡 | **Offline latent pre-computation** | `tools/offload_data/get_short-latents-giga-ctrl*.py` — see [§2.5](#25-offline-latent-pre-computation--conversion) |
| 🔴 | **📊 WMBench benchmark** | Coming soon — 15 fine-grained metrics, leaderboard + VLM judging |
| 🔴 | **Full 12,980 h pre-training corpus** | The full heterogeneous pre-training dataset (Physical / Robot / Human / Giga) — coming soon |
| 🔴 | **Stage-3 RL post-training** | 3D RL post-training scripts for stronger 3D scene modeling — coming soon |
| 🔴 | **Cross-domain checkpoints (NUScenes)** | Autonomous-driving world model checkpoint — coming soon |
| 🔴 | **Long-horizon video gallery** | ♾️ 10,000+ frame continuous rollout demos — coming soon |

### Release Channels

| Channel | Purpose | Where |
| --- | --- | --- |
| 🐙 **GitHub Releases** | Tagged source snapshots with changelogs | [Releases](../../releases) |
| 🤗 **Hugging Face Model** | Giga-World-1 model weights | [GigaAI-Research/Giga-World-1](https://huggingface.co/GigaAI-Research/Giga-World-1) |
| 🤗 **Hugging Face Dataset** | Giga-World-1 toy data | [GigaAI-Research/Giga-World-1-Toydata](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata) |
| 🔷 **ModelScope Model** | ModelScope mirror for model weights | [GigaAI/Giga-World-1](https://modelscope.cn/models/GigaAI/Giga-World-1/summary) |
| 🔷 **ModelScope Dataset** | ModelScope mirror for toy data | [GigaAI/Giga-World-1-Toydata](https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata) |
| 📄 **arXiv** | Paper PDF, BibTeX | [arxiv.org/abs/2511.19861](https://arxiv.org/abs/2511.19861) |
| 🌐 **Project Page** | Videos, leaderboard, demos | [yvonne-oh.github.io/Giga-World-1-projectpage](https://yvonne-oh.github.io/Giga-World-1-projectpage) |
| 📊 **WMBench** | Public benchmark (coming soon) | *TBA* |

> 🛠️ Want a component to ship sooner? File an issue or open a PR — see [§10 FAQ & Tips](#10--faq--tips).

---

## 📑 Table of Contents

- [📰 Latest Updates](#-latest-updates)
- [📊 Open-Source Progress](#-open-source-progress)
- [🌟 Overview](#-overview)
- [✨ Key Highlights](#-key-highlights)
- [🏗️ GigaWorld-1 Architecture](#%EF%B8%8F-gigaworld-1-architecture)
- [1. 📦 Environment Setup](#1--environment-setup)
- [2. 🗃️ Data Preparation](#2-%EF%B8%8F-data-preparation)
- [3. 🚂 Training](#3--training)
- [4. 🖼️ Data & Trajectory Visualization](#4-%EF%B8%8F-data--trajectory-visualization)
- [5. 🔄 Model Merge & Checkpoint Conversion](#5--model-merge--checkpoint-conversion)
- [6. 🎬 Inference](#6--inference)
- [7. 📥 Download Models and Toy Data](#7--download-models-and-toy-data)
- [8. 🚀 Quick Start](#8--quick-start)
- [9. 📁 Repository Layout](#9--repository-layout)
- [10. ❓ FAQ & Tips](#10--faq--tips)
- [🙏 Acknowledgements](#-acknowledgements)
- [🤝 Contact](#-contact)
- [📖 Citation](#-citation)

---

## 🌟 Overview

> **A systematic study of world models as policy evaluators**, introducing WMBench and a practical design roadmap for building evaluator-oriented world models.
> 
> *— GigaAI, CVPR 2026*

Policy evaluation remains a fundamental bottleneck for robot foundation model development. Unlike large language models — which can be screened rapidly through online benchmarks — embodied policies typically require repeated real-world rollouts that are **slow, expensive, and limited by hardware and human supervision**.

This motivates using **world models as policy evaluators**: if a learned world model can faithfully preserve the relative success and failure of policy rollouts, it can substantially reduce the cost of policy iteration.

In this work, we present a systematic study of world models for robot policy evaluation and introduce **WMBench**, a benchmark built from real-robot teleoperation data and paired policy rollouts across eight manipulation task families. Using this benchmark, we study **ten video world models**, **three action representation schemes**, and more than **20,000 world-model policy rollouts** paired with real-world executions.

**GigaWorld-1** is the practical instantiation of this roadmap — a 1.3B/5B autoregressive diffusion-transformer world generator with parameter-efficient LoRA adaptation, designed for evaluator-oriented post-training rather than generic video generation.

```text
Raw robot data
    └── Data pipeline (LeRobot → GigaWorld format)
            └── Toy / public training format
                    ├── Stage-1 training (LoRA)
                    ├── Stage-2 DMD training
                    ├── LoRA / partial checkpoint merge
                    └── i2v / t2v inference
```

---

## ✨ Key Highlights

<div align="center">

| 🧊 **1.3B / 5B** Model Variants | ⚡ **>20 FPS** Flash Inference | 💾 **<24 GB** VRAM (single RTX 4090) | ♾️ **>10 min** Infinite Horizon |
| :---: | :---: | :---: | :---: |
| Nano & Pro configurations | Stage-2 DMD distillation | Consumer-grade hardware | 10,000+ frames autoregressive rollout |

</div>

- **Unified training workflow** — the same `train_*.py` + YAML setup handles both `wan2.1` (Nano) and `wan2.2_5b` (Pro)
- **Efficient LoRA fine-tuning** — default `lora_rank=128` (Stage-1) with flexible target / exclude module lists
- **Multi-term temporal memory** — `history_sizes=[16, 2, 1]` + `latent_window_size=[9]` for long-horizon consistency with short-horizon responsiveness
- **DMD2 step-distillation** — Stage-2 compresses ~20 denoising steps to 4–6, hitting **> 20 FPS on a single H20**
- **Evaluator-oriented post-training** — preserves real-robot rollout success/failure patterns rather than chasing short-horizon visual realism
- **Cross-domain transfer** — rapidly adapts to autonomous driving (NUScenes) and other domains
- **Pixel-aligned multi-view control** — head and wrist views consume view-specific Plücker / Ray Map control signals
- **Open-source toy datasets** — drop-in Stage-1 / Stage-2 training data
- **Offline-by-default W&B** — `WANDB_MODE=offline` for safer public releases
- **Merged transformer export** — one-shot LoRA + patch merge into a clean, deployment-ready transformer
- **Web visualization tool** — dual pages (URDF Viewer + Camera Calibration) with a Three.js frontend

---

## 🏗️ GigaWorld-1 Architecture

GigaWorld-1 is an **autoregressive diffusion-transformer world generator** with **parameter-efficient LoRA adaptation**, designed for evaluator-oriented post-training rather than generic video generation.

```text
┌──────────────────────────────────────────────────────────────────┐
│                      GigaWorld-1 Generator                      │
│                                                                  │
│  Prompt (T5)            First Frame (i2v) / No first frame (t2v)│
│      │                          │                                │
│      ▼                          ▼                                │
│  ┌──────────┐         ┌──────────────────────┐                   │
│  │ T5 Encode│         │  ControlNet (Plücker │                   │
│  │  → text  │         │   / ray map / depth) │                   │
│  │  embeds  │         └──────────┬───────────┘                   │
│  └────┬─────┘                    │                                │
│       └─────────────┐            │                                │
│                     ▼            ▼                                │
│            ┌────────────────────────────────────┐                 │
│            │   DiT Blocks + LoRA (rank=128)     │                 │
│            │   + Multi-term Memory Patch       │                 │
│            │       (short / mid / long)         │                 │
│            └─────────────────┬──────────────────┘                 │
│                              ▼                                    │
│              Latent → Wan-VAE Decode → Video                      │
│                                                                  │
│  Stage-1: LoRA pre-training / post-training                      │
│  Stage-2: DMD2 step-distillation                                 │
│  Reward (optional): VideoReward / JEPA                            │
└──────────────────────────────────────────────────────────────────┘
```

**Key design points:**

- 🎯 **Evaluator-Oriented** — preserve the relative success/failure of real robot rollouts, not just short-horizon visual realism
- 🎥 **Pixel-aligned multi-view control** — view-specific Plücker / Ray Map signals keep head & wrist views geometrically consistent over long horizons
- 🧊 **3D RL post-training** — strengthens 3D scene modeling for more faithful physical dynamics
- 🌐 **Cross-domain transfer** — rapid adaptation to autonomous driving (NUScenes) and beyond

---

## 1. 📦 Environment Setup

### 1.1 Hardware & OS

| Spec | Nano (wan2.1) | Pro (wan2.2_5b) |
| --- | --- | --- |
| Recommended VRAM (inference) | **< 24 GB** (single RTX 4090 works) | 24–48 GB (single H20 / A100) |
| Recommended VRAM (training) | 4 × 24 GB minimum (ZeRO-2) | 8 × 48 GB minimum (ZeRO-2/3) |
| OS | Linux (verified on Ubuntu 20.04 / 22.04) | Same |
| CUDA | Matches your local PyTorch (CUDA 12.x recommended) | Same |
| Disk | ≥ 80 GB for inference; ≥ 500 GB NVMe for training | ≥ 1 TB NVMe |

### 1.2 Install Dependencies

The repository expects to reuse an existing Conda environment (by default `<PATH_TO_ENV>`, e.g. `~/envs/Helios`). You can point `install.sh` at any other env via the `DEFAULT_ENV_PATH` variable at the top of the script. `install.sh` auto-detects and activates that environment, then installs all dependencies.

```bash
cd <PROJECT_ROOT>
bash install.sh
```

[CODE0](./install.sh) workflow:

1. Source `miniconda3/etc/profile.d/conda.sh` if it exists
2. `conda activate <PATH_TO_ENV>`  (e.g. `conda activate /path/to/your/env`)
3. `pip install --upgrade pip setuptools wheel`
4. `pip install -r requirements.txt`
5. If `thirdparty/diffusers` exists, `pip install -e ./thirdparty/diffusers` (editable install — required for custom diffusers modifications)
6. If `thirdparty/flash-attention-3` exists, **print a notice only — do not auto-compile** (depends on your CUDA / PyTorch version)

Main dependencies (see [CODE0](./requirements.txt)):

```text
accelerate>=1.1.0        # accelerate launch / DDP / DeepSpeed
av>=12.0.0               # video read / write
decord>=0.6.0            # fast video decoding
diffusers>=0.35.0        # custom-modified version (see thirdparty/)
einops>=0.8.0
imageio>=2.36.0
imageio-ffmpeg>=0.5.1
numpy>=1.24,<3
omegaconf>=2.3.0
opencv-python>=4.9.0
packaging>=24.0
pandas>=2.1.0
peft>=0.12.0             # LoRA implementation
Pillow>=10.0.0
pyyaml>=6.0.0
safetensors>=0.4.5
torchdata>=0.8.0
tqdm>=4.66.0
transformers>=4.45.0
wandb>=0.18.0            # offline by default
xformers>=0.0.28.post3   # memory-efficient attention
```

> **Optional: build flash-attention-3 manually** (only if you need FA3)
> 
> ```bash
> cd thirdparty/flash-attention-3
> # Follow its README
> ```

### 1.3 Released Model Weights

Released weights are available from:

- Hugging Face: [GigaAI-Research/Giga-World-1](https://huggingface.co/GigaAI-Research/Giga-World-1)
- ModelScope: [GigaAI/Giga-World-1](https://modelscope.cn/models/GigaAI/Giga-World-1/summary)

| Stage | Name | Path | Notes |
| --- | --- | --- | --- |
| Open Source | WAN 2.1 1.3B FunControl | [Wan2.1-Fun-1.3B-Control](https://www.bing.com/search?q=WAN+2.1+1.3B+FunControl&form=ANNTH1&refig=6a436fa312dd427c9dfef065aca873ed&pc=CNNDDB) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&logoColor=black) | Open-source model. |
| Open Source | WAN 2.2 5B FunControl | [Wan2.2-Fun-5B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&logoColor=black) | Open-source model. |
| Before Stage 1 | GigaRobo Alpha Diffusers | `before_stage1/Wan2p1_1p3B-FunContro-GigaRobo-alpha-diffusers/` | Pretrained on Giga dataset, then converted to Diffusers. |
| Before Stage 1 | WAN 2.1 1.3B Diffusers | `before_stage1/Wan2p1_1p3B-FunControl-diffusers/` | Vanilla Diffusers-converted checkpoint. |
| Before Stage 1 | WAN 2.2 5B Diffusers | `before_stage1/Wan2p2_5B-FunControl-diffusers/` | Vanilla Diffusers-converted checkpoint. |
| Stage 1 | Nano (1.3B) | `stage1/nano/` | Stage-1 fine-tuned from the 1.3B branches. |
| Stage 1 | Pro (5B) | `stage1/pro/` | Stage-1 fine-tuned from the 5B branch. |
| Stage 2 | Nano Distill | 🚧 Coming soon | 🚧 Coming soon. |
| Stage 2 | Pro Distill | 🚧 Coming soon | 🚧 Coming soon. |

Each Stage-1 variant (`nano` / `pro`) contains two artifacts: a full Diffusers-format checkpoint and a scene LoRA checkpoint.

```text
stage1/{nano,pro}/
├── Giga-World-1-*-stage1_final-diffusers/         # full Diffusers checkpoint
│   ├── model_index.json                           # Diffusers pipeline index
│   ├── transformer/                               # DiT / video transformer weights
│   ├── vae/                                       # VAE weights
│   ├── text_encoder/                              # text encoder weights
│   ├── tokenizer/                                 # tokenizer files
│   ├── scheduler/                                 # scheduler config
│   ├── image_encoder/                             # image encoder weights
│   └── image_processor/                           # image preprocessing config
└── Giga-World-1-*-stage1_scene_lora/              # scene LoRA checkpoint
    ├── pytorch_lora_weights.safetensors           # LoRA weights for inference
    ├── transformer_full/                          # full transformer export
    ├── transformer_partial.pth                    # partial transformer checkpoint
    ├── pytorch_model/                             # training checkpoint shards
    ├── distributed_checkpoint/                    # distributed training checkpoint
    ├── scheduler.bin                              # training scheduler state
    ├── latest                                     # latest checkpoint pointer
    ├── zero_to_fp32.py                            # ZeRO checkpoint conversion script
    └── random_states_*.pkl                        # training random states
```

### 1.4 Key Environment Variables

The inference and training scripts export a set of HuggingFace / NCCL / Vulkan-friendly variables by default:

```bash
# HuggingFace
export HF_ENABLE_PARALLEL_LOADING=yes
export HF_PARALLEL_LOADING_WORKERS=8
export TOKENIZERS_PARALLELISM=false

# Attention / acceleration
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE    # skip FA compile at inference
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# NCCL (multi-GPU)
export NCCL_TIMEOUT=360000000
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TIMEOUT=360000000
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# Vulkan (prevent CPU-side errors)
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# W&B
export WANDB_MODE=offline    # default; switch to online with WANDB_API_KEY=...
```

---

## 2. 🗃️ Data Preparation

### 2.1 Data Schema

Training data in this repository is organized by **three camera views** (`cam_high` / `cam_left_wrist` / `cam_right_wrist`):

<div align="center">

<img src="assets/data_vis.gif" alt="Data Visualization" width="90%" />

</div>

```text
example/toy_datapipeline_dataset/
├── gt/                          # RGB videos (ground truth)
│   ├── cam_high/                #   head view
│   ├── cam_left_wrist/          #   left wrist view
│   └── cam_right_wrist/         #   right wrist view
├── depth/                       # Depth Anything V2 outputs
│   ├── cam_high/
│   ├── cam_left_wrist/
│   └── cam_right_wrist/
├── plucker/                     # Plücker coordinate control signals (left/right per view)
│   ├── episode_000001_left_direction.mp4
│   ├── episode_000001_left_moment.mp4
│   ├── episode_000001_right_direction.mp4
│   └── episode_000001_right_moment.mp4
├── sketch/                      # sketch control signals
│   └── cam_high/
└── labels/
    ├── data.pkl                 # per-episode metadata (see below)
    └── config.json
```

A typical record in `labels/data.pkl` (from [CODE1](./tools/datapipeline/README_datapipeline.md)):

```python
{
    "action": List[List[float]],                    # end-effector / joint actions
    "data_index": int,
    "episode_name": str,

    "cam_high_video_path": str,
    "cam_left_wrist_video_path": str,
    "cam_right_wrist_video_path": str,

    "cam_high_depth_path": str,
    "cam_left_wrist_depth_path": str,
    "cam_right_wrist_depth_path": str,

    "qpos": List[List[float]],                      # current joint angles
    "video_height": int,
    "video_width": int,
    "video_length": int,

    "short-prompt": {                                # from meta/episodes.jsonl
        "task1": {
            "start_idx": "0",
            "end_idx": "299",
            "description": "put banana into basket"
        }
    },

    "long-prompt": {                                 # generated by Qwen3-VL on cam_high
        "long prompt 1": {
            "start_idx": "0",
            "end_idx": "299",
            "caption": "The robot arm reaches toward ..."
        }
    }
}
```

### 2.2 Data Sources (≈12,980 hours)

GigaWorld-1's pre-training corpus is composed of four complementary sources (figures from the [project page](https://github.com/Yvonne-OH/Giga-World-1-projectpage)):

| Category | Representative Sources | Robot Type | Hours | Modality |
| --- | --- | --- | ---: | --- |
| 🌐 Physical | Internet Videos, Physics Videos | ⚙️ N/A | ~1,298 | 🎥 RGB Video |
| 🤖 Robot | Open X, AgiBot | 🦿 Single-arm / 🤖 Dual-arm / 🧍 Humanoid | ~5,377 | 🦾 Robot Demonstration |
| ✋ Human | EgoDex, SynData | ✋ Human Hands | ~2,411 | 🎥 RGB Video + 🖐️ Hand Pose |
| 🧠 Giga | Giga Humanoid, Giga Dual-arm | 🧍 Humanoid / 🤖 Dual-arm | ~3,894 | 🦾 Robot Demonstration |
| **🟦 Total** |   |   | **≈ 12,980 h** |   |

> This repository ships only the toy subset for reproducible verification; large-scale data should be processed following the steps in this section.

### 2.3 Use the Bundled Toy Datasets

Toy data is available from:

- Hugging Face: [GigaAI-Research/Giga-World-1-Toydata](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata)
- ModelScope: [GigaAI/Giga-World-1-Toydata](https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata)

The toy data package contains inference assets, a raw LeRobot-format toy dataset, and preprocessed model training data:

```text
example/
├── infer_assest/                # inference / rollout assets
│   ├── control_video.mp4
│   └── input_image.png
├── toy_datapipeline_dataset/    # raw LeRobot-format toy dataset
│   ├── gt/                      # RGB videos
│   ├── depth/                   # Depth Anything V2 outputs
│   ├── plucker/                 # Plücker coordinate control signals
│   ├── sketch/                  # sketch control signals
│   └── labels/                  # data.pkl + config.json
└── toy_train_dataset/           # model training data
    ├── nano/
    │   ├── dataset_cache.pkl
    │   └── episode_*.pt
    └── pro/
        ├── dataset_cache.pkl
        └── episode_*.pt
```

`example/toy_train_dataset/` is already organized in the training format, referenced by:

- [CODE0](./scripts/training/configs/stage_1_post_functrl_wan21.yaml#L21-L22) → `example/toy_train_dataset/nano`
- [CODE0](./scripts/training/configs/stage_1_post_functrl_wan22_5b.yaml#L21-L22) → `example/toy_train_dataset/pro`
- [CODE0](./scripts/training/configs/stage_2_dmd_functrl_wan21.yaml#L23) → `example/toy_train_dataset/nano`

Use them directly — no extra preprocessing required.

### 2.4 LeRobot Raw-Data Preprocessing Pipeline

Main script: [CODE0](./tools/datapipeline/datapipeline_lerobot.py)

Expected input layout (LeRobot-style):

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

Run with a single command:

```bash
python tools/datapipeline/datapipeline_lerobot.py \
  --output_base output \
  --data_dir_list origin_data/task1 \
  --num_gpus 8 \
  --max_tasks -1
```

The pipeline will:

1. Read episode metadata from `data/chunk-*/episode_*.parquet`
2. Read the three camera-view RGB videos
3. Load short task descriptions from `meta/episodes.jsonl`
4. Use **Qwen3-VL** on the `cam_high` view to generate dense long captions
5. Run **Depth Anything V2** on all three views to produce depth videos
6. Write the final dataset: `gt/`, `depth/`, `labels/data.pkl`, `labels/config.json`, `config.json`

Key constants at the top of the script that you may need to tune:

| Variable | Default | Description |
| --- | --- | --- |
| `VIDEO_FPS` | 30 | Output video FPS |
| `TARGET_HEIGHT` | 480 | Output video height |
| `TARGET_WIDTH` | 640 | Output video width |
| `CAPTION_MODEL_PATH` | local Qwen3-VL path | Caption model checkpoint |
| `CAPTION_MAX_PIXELS` | `360 * 420` | Video tokenization limit |
| `CAPTION_FPS` | 2.0 | Sampling FPS sent to the VLM |
| `CAPTION_MAX_NEW_TOKENS` | 256 | Max caption generation length |
| `LONG_PROMPT_SEGMENT_FRAMES` | 300 | Frames per long-caption segment |

> 📖 Full field reference: [CODE0](./tools/datapipeline/README_datapipeline.md).

### 2.5 Offline Latent Pre-computation & Conversion

To reduce I/O overhead during training, you can pre-encode latents before training:

- [CODE0](./tools/offload_data/get_short-latents-giga-ctrl.py)
- [CODE0](./tools/offload_data/get_short-latents-giga-ctrl-wan22-5b.py)
- [CODE0](./tools/offload_data/gigactrl2helios.py)

Checkpoint key normalization: [CODE0](./tools/others/convert_ckpt.py)

---

## 3. 🚂 Training

Training entrypoints and launcher scripts are paired (each pair = one `accelerate launch` command + one YAML).

### 3.1 Stage-1 Training (Controllable Pre-training)

| Entrypoint / Config | Launcher | Note |
| --- | --- | --- |
| Trainer: [CODE0](./train_gigaworld_functrl_uni_stage1.py) |   | Unified trainer handling both Nano and Pro |
| Config: [CODE0](./scripts/training/configs/stage_1_post_functrl_wan21.yaml) | [CODE0](./scripts/training/stage1/train_deepspeed_stage1_functrl_wan21.sh) | Nano (1.3B) |
| Config: [CODE0](./scripts/training/configs/stage_1_post_functrl_wan22_5b.yaml) | [CODE0](./scripts/training/stage1/train_deepspeed_stage1_functrl_wan22_5b.sh) | Pro (5B) |

**Launch Nano**:

```bash
bash scripts/training/stage1/train_deepspeed_stage1_functrl_wan21.sh
```

**Launch Pro**:

```bash
bash scripts/training/stage1/train_deepspeed_stage1_functrl_wan22_5b.sh
```

**Key config fields** (using `stage_1_post_functrl_wan21.yaml` as the example):

```yaml
data_config:
  single_res: true
  single_height: 480
  single_width: 1920            # width after the three views are concatenated
  dataloader_num_workers: 8
  caption_dropout_p: 0
  instance_data_root:
    - "example/toy_train_dataset/nano"

model_config:
  model_type: "wan2.1"
  pretrained_model_name_or_path: "...Wan2.1-Fun-V1.1-1.3B-giga-ctrl-2200"
  transformer_model_name_or_path: "...Wan2.1-Fun-V1.1-1.3B-giga-ctrl-2200"
  lora_rank: 128
  lora_alpha: 128.0
  lora_layers: "all-linear"     # LoRA applied to all Linear layers (excluding norm)
  lora_exclude_modules: [down, up]   # no LoRA on down/up sampling
  is_control_model: true

training_config:
  max_train_steps: 1000000
  train_batch_size: 1
  gradient_accumulation_steps: 1
  checkpointing_steps: 500
  learning_rate: 3e-5
  lr_scheduler: "constant"
  lr_warmup_steps: 500
  optimizer: "adamw"
  mixed_precision: "bf16"
  allow_tf32: true
  gradient_checkpointing: true
  enable_xformers_memory_efficient_attention: true
  # Multi-term memory + sliding window
  history_sizes: [16, 2, 1]     # 16 / 2 / 1-frame long-term context
  latent_window_size: [9]       # latent window per forward pass
  # Anti-drift / anti-forgetting
  is_random_drop: true
  random_drop_v2v_ratio: 0.4
  random_drop_t2v_ratio: 0.4
  corrupt_history: true
  corrupt_mode_history: "noise"
  # Validation cadence
  validation_steps: 500
```

Default output layout:

```text
output/
├── exp/
│   ├── Giga-world-Nano-Train-Stage-1/
│   └── Giga-world-Pro-Train-Stage-1/
└── logs/
```

### 3.2 Stage-2 DMD Training (Acceleration Distillation)

| Entrypoint / Config | Launcher |
| --- | --- |
| Trainer: [CODE0](./train_gigaworld_functrl_uni_stage2_dmd.py) |   |
| Config: [CODE0](./scripts/training/configs/stage_2_dmd_functrl_wan21.yaml) | [CODE0](./scripts/training/stage2/train_deepspeed_stage2_functrl_wan21.sh) |
| Config: [CODE0](./scripts/training/configs/stage_2_dmd_functrl_wan22_5b.yaml) | [CODE0](./scripts/training/stage2/train_deepspeed_stage2_functrl_wan22_5b.sh) |

DMD2 compresses the denoising loop from 20 steps to **4–6 steps** (the Stage-2 config sets `num_inference_steps: 6`) and aligns with a frozen real score model via a `critic_lora`.

**Launch Nano DMD**:

```bash
bash scripts/training/stage2/train_deepspeed_stage2_functrl_wan21.sh
```

Example output:

```text
output/exp/Giga-world-Nano-Train-DMD/
```

### 3.3 Multi-GPU / DeepSpeed Configuration

The launcher scripts auto-detect the number of visible GPUs via `nvidia-smi -L` and launch with DeepSpeed ZeRO-2:

- [CODE0](./scripts/accelerate_configs/example_zero2.yaml)
- [CODE0](./scripts/accelerate_configs/example_zero3.yaml)
- [CODE0](./scripts/accelerate_configs/zero2.json)
- [CODE0](./scripts/accelerate_configs/zero3.json)

> The launchers set `NCCL_TIMEOUT` / `TORCH_NCCL_BLOCKING_WAIT` and friends to keep long runs from being kicked out. The EMA ZeRO-3 port is adjustable via `ema_zero3_port` in the config.

---

## 4. 🖼️ Data & Trajectory Visualization

[CODE0](./tools/data_vis_tools/README.md) ships a dual-page web tool:

| Page | URL | Description |
| --- | --- | --- |
| 🦾 URDF 3D Viewer | `http://127.0.0.1:8090/` | Load pkl → parse `action` / `qpos` → 3D bimanual URDF/STL animation |
| 📷 Camera Calibration | `http://127.0.0.1:8090/calib` | Intrinsics/extrinsics visualization + multi-frame overlay + 3D FK + camera projection |

```bash
cd tools/data_vis_tools
python app.py --host 0.0.0.0 --port 8090
```

**Feature highlights:**

- **6 joint angles per arm** with live updates and draggable sliders
- **WebM recording** of the current 3D view
- **Multi-frame overlay** on the calibration page (configurable frame step) for projection-consistency inspection
- **Normalized joint curves** at the bottom-right showing min-max-normalized left/right arm joints over time, with a vertical line marking the current frame

---

## 5. 🔄 Model Merge & Checkpoint Conversion

Unified merge tool: [CODE0](./tools/ckpt_tools/uni_merge_lora_for_giga_world_1.py)

Supports both `wan2.1` and `wan2.2_5b`; auto-resolves LoRA and partial state dicts from a checkpoint directory; the output is a stand-alone, deployment-ready transformer.

```bash
python tools/ckpt_tools/uni_merge_lora_for_giga_world_1.py \
  --base_model <PATH_TO_BASE_NANO> \
  --save_dir   <PATH_TO_STAGE1_MERGED_NANO> \
  --ckpt_dir   /path/to/checkpoint-XXXX \
  --model_type wan2.1
```

For Pro 5B:

```bash
python tools/ckpt_tools/uni_merge_lora_for_giga_world_1.py \
  --base_model <PATH_TO_BASE_PRO> \
  --save_dir   <PATH_TO_STAGE1_MERGED_PRO> \
  --ckpt_dir   /path/to/checkpoint-XXXX \
  --model_type wan2.2_5b
```

The merge process also writes a **visual HTML report** at `<save_dir>/merge_report.html`, recording the source, merge method, and success status of every part — handy for release traceability.

**Auxiliary conversion tools:**

- Key rename / normalization: [CODE0](./tools/others/convert_ckpt.py)
- Data-layout migration / pre-computation: [CODE0](./tools/offload_data/gigactrl2helios.py)

---

## 6. 🎬 Inference

### 6.1 One-Click Scripts (i2v / t2v × Nano / Pro)

| Script | Mode | Model | Link |
| --- | --- | --- | --- |
| `run_infer_nano_i2v.sh` | i2v | Nano 1.3B | [script](./scripts/infer/run_infer_nano_i2v.sh) |
| `run_infer_nano_t2v.sh` | t2v | Nano 1.3B | [CODE0](./scripts/infer/run_infer_nano_t2v.sh) |
| `run_infer_pro_i2v.sh` | i2v | Pro 5B | [script](./scripts/infer/run_infer_pro_i2v.sh) |
| `run_infer_pro_t2v.sh` | t2v | Pro 5B | [CODE0](./scripts/infer/run_infer_pro_t2v.sh) |

Usage:

```bash
# Nano i2v: first frame + control video + text prompt
bash scripts/infer/run_infer_nano_i2v.sh

# Pro t2v: text prompt only (omit --image_path → t2v mode)
bash scripts/infer/run_infer_pro_t2v.sh
```

Output location (created by the script via `mkdir -p`):

```text
output/infer_results/
├── giga_i2v_nano/
├── giga_t2v_nano/
├── giga_i2v_pro/
└── giga_t2v_pro/
```

Output videos are saved at **10 FPS** by default.

### 6.2 Command-Line Arguments

The underlying entrypoint [CODE0](./infer/infer_giga_world.py) exposes the following arguments:

| Argument | Required | Default | Description |
| --- | :---: | --- | --- |
| `--config` | ✅ | — | Training / inference YAML config (drives model type and hyperparams) |
| `--base_model_path` | ✅ | — | Base diffusers model directory (VAE / T5 / Transformer) |
| `--transformer_model_name_or_path` |   | None | Path to the merged transformer; falls back to `--base_model_path` if None |
| `--checkpoint_path` |   | None | Optional LoRA / partial checkpoint path |
| `--image_path` |   | None | **First frame for i2v**; omit to enter **t2v mode** |
| `--prompt` | ✅ | — | Text prompt |
| `--control_video_path` |   | None | Control video (Plücker / Ray Map), optional |
| `--output_dir` | ✅ | — | Output root directory |
| `--sample_name` |   | sample | Output video name prefix |
| `--seed` |   | 42 | Random seed |
| `--fps` |   | 24 | Output video FPS |
| `--num_frames` |   | 99 | Total frames (330 ≈ 33 s @ 10 FPS) |
| `--height` |   | 480 | Output height |
| `--width` |   | 1920 | Output width (typically 640×3 = 1920 for three views) |
| `--num_inference_steps` |   | 20 | 20 steps for Stage-1; 4–6 for Stage-2 / DMD |
| `--guidance_scale` |   | 5.0 | Classifier-free guidance strength |
| `--enable_tiling` |   | False | VAE tiling for memory savings |

### 6.3 Inference Output Example

<div align="center">

| First Frame | Control Video | Generated Rollout |
| :---: | :---: | :---: |
| ![input](example/infer_assest/input_image.png) | ▶ [CODE0](./example/infer_assest/control_video.mp4) | *(produced under `output/infer_results/`)* |

</div>

> In i2v mode, the model uses [CODE0](./example/infer_assest/input_image.png) as the first frame and consumes the Plücker / Ray Map control signal from `control_video.mp4` in the same directory.

For richer visual results, see the project page:

- 🦾 **Multi-view control GIF grid** — [Giga-World-1-projectpage/video/control_gif](https://github.com/Yvonne-OH/Giga-World-1-projectpage/tree/main/video/control_gif)
- ♾️ **Long-horizon rollout demo** — [Giga-World-1-projectpage/video/flash_and_ultra_gen](https://github.com/Yvonne-OH/Giga-World-1-projectpage/tree/main/video/flash_and_ultra_gen)
- ✅❌ **Closed-loop rollout comparison** — [Giga-World-1-projectpage/video/cc](https://github.com/Yvonne-OH/Giga-World-1-projectpage/tree/main/video/cc)

---

## 7. 📥 Download Models and Toy Data

Use the one-click downloader in [CODE0](./tools/download_tool/README.md) to download released model weights and toy data from Hugging Face or ModelScope.

Supported release URLs:

| Platform | Target | URL |
| --- | --- | --- |
| Hugging Face | Model | [GigaAI-Research/Giga-World-1](https://huggingface.co/GigaAI-Research/Giga-World-1) |
| Hugging Face | Toy data | [GigaAI-Research/Giga-World-1-Toydata](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata) |
| ModelScope | Model | [GigaAI/Giga-World-1](https://modelscope.cn/models/GigaAI/Giga-World-1/summary) |
| ModelScope | Toy data | [GigaAI/Giga-World-1-Toydata](https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata) |

Download from Hugging Face:

```bash
python tools/download_tool/download_giga_world.py \
  --platform hf \
  --target all \
  --output-dir ./downloads
```

Download from ModelScope:

```bash
python tools/download_tool/download_giga_world.py \
  --platform modelscope \
  --target all \
  --output-dir ./downloads
```

You can also download only one target:

```bash
# model only
python tools/download_tool/download_giga_world.py --platform hf --target model --output-dir ./downloads

# toy data only
python tools/download_tool/download_giga_world.py --platform hf --target toydata --output-dir ./downloads
```

Downloaded files are saved as:

```text
./downloads/
├── Giga-World-1/              # model weights
└── Giga-World-1-Toydata/      # toy data
```

## 8. 🚀 Quick Start

In about 30 minutes you can go from a fresh clone to a first inference:

```bash
# 0) Set your local path variables (adjust to your machine)
export PROJECT_ROOT="$(pwd)"             # the directory you cloned this repo into
export BASE_NANO="<PATH_TO_BASE_NANO>"   # e.g. ./downloads/Giga-World-1/before_stage1/Wan2p1_1p3B-FunControl-diffusers
export STAGE1_NANO="<PATH_TO_STAGE1_MERGED_NANO>"  # e.g. ./downloads/Giga-World-1/stage1/nano/Giga-World-1-nano-stage1_final-diffusers

# 1) Install dependencies (auto-activates the configured Conda env)
cd "${PROJECT_ROOT}"
bash install.sh

# 2) Run a Nano Stage-1 toy training (auto-uses example/toy_train_dataset/nano)
bash scripts/training/stage1/train_deepspeed_stage1_functrl_wan21.sh

# 3) After some steps, merge LoRA + patch into a stand-alone transformer
python tools/ckpt_tools/uni_merge_lora_for_giga_world_1.py \
  --base_model "${BASE_NANO}" \
  --save_dir   "${STAGE1_NANO}" \
  --ckpt_dir   "${PROJECT_ROOT}/output/exp/Giga-world-Nano-Train-Stage-1/checkpoint-500" \
  --model_type wan2.1

# 4) Run an i2v inference with the merged transformer
bash scripts/infer/run_infer_nano_i2v.sh
```

> For **Pro 5B** replace the corresponding items in 1)/3)/4):
> 
> - Config: `stage_1_post_functrl_wan22_5b.yaml`
> - Launcher: `scripts/training/stage1/train_deepspeed_stage1_functrl_wan22_5b.sh`
> - Merge with `--model_type wan2.2_5b`
> - Inference: `scripts/infer/run_infer_pro_i2v.sh`

---

## 9. 📁 Repository Layout

```text
.
├── gigaworld/                         # Core model, pipeline, data loader, scheduler, and utils
│   ├── dataset/                       #   Stage-1 / Stage-2 / DMD data loaders
│   ├── modules/                       #   Transformer + custom Triton / Flash kernels
│   │   ├── gigaworld_kernels/         #     fp32_rmsnorm, tiled_linear, triton_norm, triton_rope
│   │   ├── transformer_gigaworld.py
│   │   └── transformer_functrl_gigaworld.py
│   ├── pipelines/                     #   i2v / t2v main pipelines
│   ├── scheduler/                     #   custom schedulers
│   ├── utils/                         #   TrainConfig, EMA, recycle batch, etc.
│   └── videoalign/                    #   reward / VLM training & inference
├── infer/
│   └── infer_giga_world.py            # Python inference entrypoint
├── scripts/
│   ├── accelerate_configs/            # DeepSpeed ZeRO-2 / ZeRO-3 configs
│   ├── infer/                         #   i2v / t2v × Nano / Pro one-click scripts
│   ├── training/
│   │   ├── configs/                   #   Stage-1 / Stage-2 YAMLs
│   │   ├── stage1/                    #   Nano / Pro Stage-1 launchers
│   │   └── stage2/                    #   Nano / Pro Stage-2 DMD launchers
├── tools/
│   ├── ckpt_tools/                    #   LoRA merge and checkpoint utilities
│   ├── datapipeline/                  #   LeRobot-style data preprocessing
│   ├── data_vis_tools/                #   Web URDF + camera calibration viewer
│   ├── download_tool/                 #   one-click HF / ModelScope downloader
│   ├── offload_data/                  #   offline latent pre-computation / format conversion
│   └── others/                        #   misc conversion tools
├── assets/                            #   project page hero / teaser media
│   ├── main_page.png                  #   main teaser image
│   └── data_vis.gif                   #   data visualization demo
├── example/
│   ├── infer_assest/                  #   example first frame + control video
│   ├── toy_train_dataset/             #   Nano / Pro toy training datasets
│   └── toy_datapipeline_dataset/      #   toy preprocessing output (gt / depth / plucker / sketch)
├── model/
│   ├── before_stage1/                 #   Diffusers-converted base checkpoints
│   ├── stage1/                        #   Nano / Pro Stage-1 checkpoints
│   └── stage2_distill/                #   distilled checkpoints (coming soon)
├── train_gigaworld_functrl_uni_stage1.py
├── train_gigaworld_functrl_uni_stage2_dmd.py
├── requirements.txt
└── install.sh
```

---

## 10. ❓ FAQ & Tips

- **Q: Do I need to rewrite all the absolute paths in the YAMLs?**
A: Yes. `pretrained_model_name_or_path`, `transformer_model_name_or_path`, `real_score_model_name_or_path`, and `reward_model_name_or_path` are all hard-coded for the original machine. Replace them with paths matching your local `mnt / shared_disk` layout, as listed in Section 1.3.

- **Q: How do I switch W&B to online mode?**
A: Before launching, set `export WANDB_MODE=online` and `export WANDB_API_KEY=...`. The default `offline` mode writes logs locally without uploading.

- **Q: I2V motion is very slow at the beginning — what should I do?**
A: See the comments in [CODE0](./scripts/training/configs/correct.yaml). Adding the "first-frame + last-frame anchor" data format during training significantly mitigates this. The same YAML also enables the anti-drift switches `corrupt_mode_history: "random"` and `is_add_saturation: true`.

- **Q: Why must diffusers be installed in editable mode?**
A: This repository makes small custom modifications to diffusers' attention processor / scheduler etc. (see [CODE0](./thirdparty/diffusers)). `pip install -e` is required for those changes to be loaded.

- **Q: Stage-2 warns "real score model path should be checked"**
A: Stage-2 DMD needs a frozen real score model. The public config intentionally marks this field as `FIXME`; point `real_score_model_name_or_path` to your Stage-1 merged transformer.

- **Q: Can I run this on consumer GPUs?**
A: Nano (1.3B) + Stage-2 DMD fits in a single **< 24 GB** card (RTX 4090 works). Pro 5B needs at least 24 GB, ideally 48 GB.

- **Q: The visualization tool won't open?**
A: The default port is 8090. If it is occupied, run `python app.py --port 8091`. Use `http://<host>:8090/calib` to switch to the camera calibration page.

---

## 🙏 Acknowledgements

GigaWorld-1 stands on the shoulders of a vibrant open-source ecosystem. We are deeply grateful to the following communities and projects that made this work possible:

### 🤗 Foundation Models & Architectures
- [**Wan (Alibaba)**](https://github.com/Wan-Video/Wan2.1) — the `wan2.1` and `wan2.2_5b` backbones that power GigaWorld-1 Nano and Pro
- [**Diffusers**](https://github.com/huggingface/diffusers) — the modular diffusion framework we extend with custom attention processors and pipelines
- [**Hugging Face 🤗**](https://huggingface.co/) — hosting, `transformers`, `accelerate`, and the entire model & dataset ecosystem
- [**GigaAI-Research/Giga-World-1**](https://huggingface.co/GigaAI-Research/Giga-World-1) and [**GigaAI-Research/Giga-World-1-Toydata**](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata) — the public Hugging Face model and toy-data repositories

### 🛠️ Training & Acceleration
- [**PyTorch**](https://pytorch.org/) & [**DeepSpeed**](https://www.deepspeed.ai/) — the foundation of our distributed training stack
- [**Accelerate**](https://github.com/huggingface/accelerate) — the launcher that ties everything together
- [**PEFT**](https://github.com/huggingface/peft) — the LoRA implementation behind `lora_rank=128` and `critic_lora`
- [**xFormers**](https://github.com/facebookresearch/xformers) — memory-efficient attention
- [**Flash-Attention**](https://github.com/Dao-AILab/flash-attention) — fast and memory-efficient exact attention (FA2 / FA3)
- [**Triton**](https://github.com/triton-lang/triton) — custom kernels for RMSNorm, RoPE, and tiled linears (`gigaworld/modules/gigaworld_kernels/`)

### 🗃️ Data & Annotation
- [**Qwen3-VL**](https://github.com/QwenLM/Qwen3-VL) — dense long-caption generation from `cam_high` videos
- [**Depth Anything V2**](https://github.com/DepthAnything/Depth-Anything-V2) — monocular depth estimation for all three camera views
- [**LeRobot**](https://github.com/huggingface/lerobot) — the LeRobot-style raw data layout our preprocessing pipeline consumes
- [**Open X-Embodiment**](https://robotics-transformer-x.github.io/) & [**AgiBot**](https://www.agibot.com/) — large-scale robot demonstration datasets

### 🦿 Robotics, Visualization & Tooling
- [**Helios**](https://github.com/PKU-Alignment/helios) — the upstream distributed-training environment whose setup script (`helios_setup.sh`) this repo reuses via `install.sh`; thank you for paving the road on which this release travels 🚀
- [**Three.js**](https://threejs.org/) — the WebGL renderer behind the URDF Viewer and camera-calibration tool
- [**Three.js + URDFLoader**](https://github.com/gkjohnson/urdf-loaders) — URDF/STL loading and forward kinematics
- [**WandB**](https://wandb.ai/) — experiment tracking (offline-by-default in this repo)
- [**Pandas**](https://pandas.pydata.org/), [**NumPy**](https://numpy.org/), [**Pillow**](https://python-pillow.org/), [**OpenCV**](https://opencv.org/) — the daily workhorses

### 🎬 Predecessors & Inspiration
- [**VideoCrafter**](https://github.com/AILab-CVC/VideoCrafter), [**CogVideoX**](https://github.com/THUDM/CogVideo), [**Open-Sora**](https://github.com/hpcaitech/Open-Sora), [**LTX-Video**](https://github.com/Lightricks/LTX-Video) — for showing us what open-source video generation can look like
- [**Wan-Video / Fun-1.1-1.3B-InP**](https://huggingface.co/ali-vilab) — base model artifacts
- [**DMD / DMD2**](https://github.com/tianweiy/DMD2) — the diffusion distillation theory behind our Stage-2 acceleration

### 🌟 Community
A heartfelt **thank you** to everyone who has filed an issue, opened a PR, shared a workflow, or simply starred the project. Open source is a relay race — we are proud to hand the baton forward.

If you find GigaWorld-1 useful, please consider ⭐ starring the repo and citing the paper (see below).

---

## 📖 Citation

```bibtex
@article{gigaworld2025,
  title   = {GigaWorld-1: A Roadmap to World Models for Robot Policy Evaluation},
  author  = {{GigaAI}},
  journal = {arXiv preprint},
  year    = {2025},
  eprint  = {2511.19861},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

- 📄 Paper: [arxiv.org/abs/2511.19861](https://arxiv.org/abs/2511.19861)
- 🌐 Project page: [https://yvonne-oh.github.io/Giga-World-1-projectpage](https://yvonne-oh.github.io/Giga-World-1-projectpage)
- 🧑‍💻 Project repo: [https://github.com/Yvonne-OH/Giga-World-1-projectpage](https://github.com/Yvonne-OH/Giga-World-1-projectpage)
- 🤗 Model: [https://huggingface.co/GigaAI-Research/Giga-World-1](https://huggingface.co/GigaAI-Research/Giga-World-1)
- 🤗 Toy data: [https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata)
- 🔷 ModelScope model: [https://modelscope.cn/models/GigaAI/Giga-World-1/summary](https://modelscope.cn/models/GigaAI/Giga-World-1/summary)
- 🔷 ModelScope toy data: [https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata](https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata)

---

<div align="center">

<sub>Built with ❤️ by the <b>GigaWorld Team, GigaAI</b> · CVPR 2026</sub>

<sub>Released under the Apache 2.0 License.</sub>

</div>
