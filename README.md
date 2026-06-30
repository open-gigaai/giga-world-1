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

## 📑 Table of Contents

- [📰 Latest Updates](#-latest-updates)
- [📊 Open-Source Progress](#-open-source-progress)

- [1. 📦 Environment Setup](#1--environment-setup)
- [2. 🗃️ Data Preparation](#2-%EF%B8%8F-data-preparation)
- [3. 🧩 Model Preparation](#3--model-preparation)
- [4. 🚂 Training](#4--training)
- [5. 🎬 Inference](#5--inference)
- [6. 🔄 Model Merge & Checkpoint Conversion](#6--model-merge--checkpoint-conversion)
- [7. 📁 Repository Layout](#7--repository-layout)
- [🙏 Acknowledgements](#-acknowledgements)
- [📖 Citation](#-citation)

---

## 📰 Latest Updates

<div align="center">

| Date | Update |
| :---: | --- |
| 🧑‍💻 **2026-07** | Partial training, inference, data processing, and model utility code was open-sourced. |
| 📦 **2026-07** | Partial model weights, toy data, and download tools were released. |
| 📖 **2026-07** | The GigaWorld-1 technical report was released. |
| 🏆 **2026-03** | We hosted the CVPR 2026 World Model Challenge. See [CVPR-2026-Workshop-WM-Track](https://github.com/open-gigaai/CVPR-2026-Workshop-WM-Track/tree/main/). |

</div>

> 💡 **Subscribe to releases** — click **Watch ▾ → Custom → Releases** on the GitHub repo to be notified when new weights, datasets, or the WMBench benchmark drop.

---

## 📊 Open-Source Progress

> 🟢 **Released** · 🟡 **Beta** · 🔴 **Coming Soon** — last updated 2026-07

| Status | Component | Description |
| :---: | --- | --- |
| 🟢 | **Stage-1 weights (Nano / Pro)** | Released on [GigaAI-Research/Giga-World-1](https://huggingface.co/GigaAI-Research/Giga-World-1) and [ModelScope](https://modelscope.cn/models/GigaAI/Giga-World-1/summary) |
| 🟢 | **Training code** | Stage-1: `train_gigaworld_functrl_uni_stage1.py` for Nano (1.3B) and Pro (5B), DeepSpeed ZeRO-2/3 ready — see [§3.1](#31-stage-1-training-controllable-pre-training); Stage-2: `train_gigaworld_functrl_uni_stage2_dmd.py` for DMD2 distillation (4–6 steps) — see [§3.2](#32-stage-2-dmd-training-acceleration-distillation) |
| 🟢 | **Inference code (i2v / t2v)** | Nano + Pro one-click scripts, 10 FPS, 33 s rollouts — see [§5](#5--inference) |
| 🟡 | **Data preprocessing pipeline & toy data** | LeRobot-style → GigaWorld format with Qwen3-VL captions + Depth Anything V2 — see [§2.4](#24-lerobot-raw-data-preprocessing-pipeline); toy data: [GigaAI-Research/Giga-World-1-Toydata](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata) |
| 🟢 | **Tools** | LoRA merge / checkpoint conversion, visualization, and offline latent utilities — see [§4](#4-%EF%B8%8F-data--trajectory-visualization), [§6](#6--model-merge--checkpoint-conversion), [§2.5](#25-offline-latent-pre-computation--conversion) |
| 🔴 | **📊 WMBench benchmark** | Coming soon — 15 fine-grained metrics, leaderboard + VLM judging |
| 🔴 | **Stage-2 distilled weights** | Distilled Nano / Pro checkpoints — coming soon |
| 🔴 | **RL post-training** | 3D RL post-training scripts for stronger 3D scene modeling — coming soon |
| 🔴 | **Other-domain weights and training code** | Additional domain checkpoints and corresponding training recipes — coming soon |
| 🔴 | **Acceleration framework** | Optimized distributed inference / training acceleration stack — coming soon |

### 🌐 Release Channels

| Channel | Purpose | Where |
| --- | --- | --- |
| 🐙 **GitHub Releases** | Tagged source snapshots with changelogs | <a href="../../releases"><img src="https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub Repository"></a> |
| 🤗 **Hugging Face Model** | Giga-World-1 model weights | <a href="https://huggingface.co/GigaAI-Research/Giga-World-1"><img src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face Model"></a> |
| 🤗 **Hugging Face Dataset** | Giga-World-1 toy data | <a href="https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face Dataset"></a> |
| 🔷 **ModelScope Model** | ModelScope mirror for model weights | <a href="https://modelscope.cn/models/GigaAI/Giga-World-1/summary"><img src="https://img.shields.io/badge/ModelScope-Model-624AFF?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMiAxMiI+PHJlY3Qgd2lkdGg9IjEyIiBoZWlnaHQ9IjEyIiByeD0iMiIgZmlsbD0iIzYyNEFGRiIvPjwvc3ZnPg%3D%3D&logoColor=white" alt="ModelScope Model"></a> |
| 🔷 **ModelScope Dataset** | ModelScope mirror for toy data | <a href="https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata"><img src="https://img.shields.io/badge/ModelScope-Dataset-624AFF?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMiAxMiI+PHJlY3Qgd2lkdGg9IjEyIiBoZWlnaHQ9IjEyIiByeD0iMiIgZmlsbD0iIzYyNEFGRiIvPjwvc3ZnPg%3D%3D&logoColor=white" alt="ModelScope Dataset"></a> |
| 📄 **arXiv** | Paper PDF, BibTeX | <a href="https://arxiv.org/abs/2511.19861"><img src="https://img.shields.io/badge/arXiv-2511.19861-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv Paper"></a> |
| 🌐 **Project Page** | Videos, leaderboard, demos | <a href="https://yvonne-oh.github.io/Giga-World-1-projectpage">🌐 Project Page</a> |
| 📊 **WMBench** | Public benchmark (coming soon) | *TBA* |
| 🆘 **Support** | Issues and discussions | <a href="../../issues"><img src="https://img.shields.io/badge/GitHub-Issues-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub Issues"></a> |

> 🛠️ Want a component to ship sooner? File an issue or open a PR — see [§10 FAQ & Tips](#10--faq--tips).

---

## 1. 📦 Environment Setup

### 1.1 Hardware & OS

| Item | Requirement / Recommendation |
| --- | --- |
| Production setup | Single node with **8 × H20** or **8 × A100** GPUs |
| Inference | Supports both Nano (1.3B) and Pro (5B); consumer-grade GPUs can be used with memory-saving settings |
| Training | Production experiments are run on 8-GPU nodes; consumer-grade GPU training is possible with ZeRO, offloading, gradient checkpointing, and reduced batch / resolution settings |
| OS | Linux, verified on Ubuntu 20.04 / 22.04 |
| CUDA | CUDA 12.x recommended, matching the local PyTorch installation |

> **Note:** We use a single-node 8× H20 or 8× A100 setup for production training. With appropriate memory optimization techniques, the released code can also run training and inference experiments on consumer-grade GPUs.

### 1.2 Install Dependencies

The repository expects to reuse an existing Conda environment (by default `<PATH_TO_ENV>`, e.g. `~/envs/Helios`). You can point `install.sh` at any other env via the `DEFAULT_ENV_PATH` variable at the top of the script. `install.sh` auto-detects and activates that environment, then installs all dependencies.

```bash
cd <PROJECT_ROOT>
bash install.sh
```

[install.sh](./install.sh) workflow:

1. Source `miniconda3/etc/profile.d/conda.sh` if it exists
2. `conda activate <PATH_TO_ENV>`  (e.g. `conda activate /path/to/your/env`)
3. `pip install --upgrade pip setuptools wheel`
4. `pip install -r requirements.txt`
5. If `thirdparty/diffusers` exists, `pip install -e ./thirdparty/diffusers` (editable install — required for custom diffusers modifications)
6. If `thirdparty/flash-attention-3` exists, **print a notice only — do not auto-compile** (depends on your CUDA / PyTorch version)

Main dependencies (see [requirements.txt](./requirements.txt)):

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

---

## 2. 🗃️ Data Preparation

This release provides a small toy data package for verifying inference, data loading, visualization, and training workflows. The toy data is available from [Hugging Face](https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata) and [ModelScope](https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata).

Use the one-click downloader:

```bash
bash tools/download_tool/download_giga_world.sh \
  --platform hf \
  --target toydata \
  --output-dir ./downloads
```

For ModelScope, replace `--platform hf` with `--platform modelscope`. See [tools/download_tool/README.md](./tools/download_tool/README.md) for all options.

After downloading, place or symlink the toy data under `example/`:

```bash
mkdir -p example
cp -r ./downloads/Giga-World-1-Toydata/* ./example/
```

Expected project structure:

```text
giga-world-release/
└── example/
    ├── infer_assest/                # inference / rollout assets
    ├── toy_datapipeline_dataset/    # raw LeRobot-format toy dataset
    │   ├── gt/
    │   ├── depth/
    │   ├── plucker/
    │   ├── sketch/
    │   └── labels/
    └── toy_train_dataset/           # model training data
        ├── nano/
        │   ├── dataset_cache.pkl
        │   └── episode_*.pt
        └── pro/
            ├── dataset_cache.pkl
            └── episode_*.pt
```

`toy_train_dataset/` is already in the format used by the training configs: [stage_1_post_functrl_wan21.yaml](./scripts/training/configs/stage_1_post_functrl_wan21.yaml), [stage_1_post_functrl_wan22_5b.yaml](./scripts/training/configs/stage_1_post_functrl_wan22_5b.yaml), and [stage_2_dmd_functrl_wan21.yaml](./scripts/training/configs/stage_2_dmd_functrl_wan21.yaml).

For raw data visualization, run the web tool and open `http://127.0.0.1:8090/` or `http://127.0.0.1:8090/calib`:

```bash
cd tools/data_vis_tools
python app.py --host 0.0.0.0 --port 8090
```

<p align="center">
  <img src="assets/data_vis.gif" width="90%" alt="Raw data visualization demo" />
</p>

For offline latent pre-computation, use [get_short-latents-giga-ctrl.py](./tools/offload_data/get_short-latents-giga-ctrl.py) or [get_short-latents-giga-ctrl-wan22-5b.py](./tools/offload_data/get_short-latents-giga-ctrl-wan22-5b.py). The input data should contain `helios_giga_ctrl.jsonl`, `videos/`, and `control_videos/`; outputs are `.pt` samples containing precomputed `vae_latent`, `control_latent`, `prompt_embed`, and related metadata. See [tools/offload_data/data_format.md](./tools/offload_data/data_format.md) for the data format.

## 3. 🧩 Model Preparation

Released model weights are available from:

| Platform | Repository |
| --- | --- |
| 🤗 Hugging Face | [GigaAI-Research/Giga-World-1](https://huggingface.co/GigaAI-Research/Giga-World-1) |
| 🔷 ModelScope | [GigaAI/Giga-World-1](https://modelscope.cn/models/GigaAI/Giga-World-1/summary) |

Use the download helper:

- Script: [download_giga_world.sh](./tools/download_tool/download_giga_world.sh)
- Full usage: [tools/download_tool/README.md](./tools/download_tool/README.md)

Download model weights from Hugging Face:

```bash
bash tools/download_tool/download_giga_world.sh \
  --platform hf \
  --target model \
  --output-dir ./downloads
```

Download model weights from ModelScope:

```bash
bash tools/download_tool/download_giga_world.sh \
  --platform modelscope \
  --target model \
  --output-dir ./downloads
```

After downloading, place or symlink the model files under `model/`:

```text
giga-world-release/
├── model/
│   ├── before_stage1/
│   │   ├── Wan2p1_1p3B-FunContro-GigaRobo-alpha-diffusers/
│   │   ├── Wan2p1_1p3B-FunControl-diffusers/
│   │   └── Wan2p2_5B-FunControl-diffusers/
│   ├── stage1/
│   │   ├── nano/
│   │   └── pro/
│   └── stage2_distill/          # coming soon
└── tools/
    └── download_tool/
```

If the downloader saves weights under `./downloads/Giga-World-1/`, copy or symlink them into the repository:

```bash
mkdir -p model
cp -r ./downloads/Giga-World-1/* ./model/
```

---

## 4. 🚂 Training

Training entrypoints and launcher scripts are paired (each pair = one `accelerate launch` command + one YAML).

### 3.1 Stage-1 Training (Controllable Pre-training)

| Entrypoint / Config | Launcher | Note |
| --- | --- | --- |
| Trainer: [train_gigaworld_functrl_uni_stage1.py](./train_gigaworld_functrl_uni_stage1.py) |   | Unified trainer handling both Nano and Pro |
| Config: [stage_1_post_functrl_wan21.yaml](./scripts/training/configs/stage_1_post_functrl_wan21.yaml) | [train_deepspeed_stage1_functrl_wan21.sh](./scripts/training/stage1/train_deepspeed_stage1_functrl_wan21.sh) | Nano (1.3B) |
| Config: [stage_1_post_functrl_wan22_5b.yaml](./scripts/training/configs/stage_1_post_functrl_wan22_5b.yaml) | [train_deepspeed_stage1_functrl_wan22_5b.sh](./scripts/training/stage1/train_deepspeed_stage1_functrl_wan22_5b.sh) | Pro (5B) |

**Launch Nano**:

```bash
bash scripts/training/stage1/train_deepspeed_stage1_functrl_wan21.sh
```

**Launch Pro**:

```bash
bash scripts/training/stage1/train_deepspeed_stage1_functrl_wan22_5b.sh
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
| Trainer: [train_gigaworld_functrl_uni_stage2_dmd.py](./train_gigaworld_functrl_uni_stage2_dmd.py) |   |
| Config: [stage_2_dmd_functrl_wan21.yaml](./scripts/training/configs/stage_2_dmd_functrl_wan21.yaml) | [train_deepspeed_stage2_functrl_wan21.sh](./scripts/training/stage2/train_deepspeed_stage2_functrl_wan21.sh) |
| Config: [stage_2_dmd_functrl_wan22_5b.yaml](./scripts/training/configs/stage_2_dmd_functrl_wan22_5b.yaml) | [train_deepspeed_stage2_functrl_wan22_5b.sh](./scripts/training/stage2/train_deepspeed_stage2_functrl_wan22_5b.sh) |

DMD2 compresses the denoising loop from 20 steps to **4–6 steps** (the Stage-2 config sets `num_inference_steps: 6`) and aligns with a frozen real score model via a `critic_lora`.

**Launch Nano DMD**:

```bash
bash scripts/training/stage2/train_deepspeed_stage2_functrl_wan21.sh
```

Example output:

```text
output/exp/Giga-world-Nano-Train-DMD/
```

## 5. 🎬 Inference

| Script | Mode | Model | Link |
| --- | --- | --- | --- |
| `run_infer_nano_i2v.sh` | i2v | Nano 1.3B | [script](./scripts/infer/run_infer_nano_i2v.sh) |
| `run_infer_nano_t2v.sh` | t2v | Nano 1.3B | [script](./scripts/infer/run_infer_nano_t2v.sh) |
| `run_infer_pro_i2v.sh` | i2v | Pro 5B | [script](./scripts/infer/run_infer_pro_i2v.sh) |
| `run_infer_pro_t2v.sh` | t2v | Pro 5B | [script](./scripts/infer/run_infer_pro_t2v.sh) |

Usage:

```bash
# Nano i2v: first frame + control video + text prompt
bash scripts/infer/run_infer_nano_i2v.sh

# Pro t2v: text prompt only (omit --image_path → t2v mode)
bash scripts/infer/run_infer_pro_t2v.sh
```


The underlying entrypoint [infer_giga_world.py](./infer/infer_giga_world.py) exposes the following arguments:

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
| `--fps` |   | 10 | Output video FPS |
| `--num_frames` |   | 99 | Total frames (330 ≈ 33 s @ 10 FPS) |
| `--height` |   | 480 | Output height |
| `--width` |   | 1920 | Output width (typically 640×3 = 1920 for three views) |
| `--num_inference_steps` |   | 20 | 20 steps for Stage-1; 4–6 for Stage-2 / DMD |
| `--guidance_scale` |   | 5.0 | Classifier-free guidance strength |
| `--enable_tiling` |   | False | VAE tiling for memory savings |

Inference output example:

<div align="center">

<table>
  <tr>
    <th>First Frame</th>
    <th>Control Video</th>
    <th>Generated Rollout</th>
  </tr>
  <tr>
    <td><img src="assets/input_image.png" alt="input image" width="260" /></td>
    <td><a href="assets/control_video.mp4"><img src="assets/input_image.png" alt="control video" width="260" /><br />control_video.mp4</a></td>
    <td><a href="assets/i2v_sample.mp4"><img src="assets/input_image.png" alt="generated rollout" width="260" /><br />i2v_sample.mp4</a></td>
  </tr>
</table>

</div>

---

## 6. 🔄 Model Merge & Checkpoint Conversion

Unified merge tool: [uni_merge_lora_for_giga_world_1.py](./tools/ckpt_tools/uni_merge_lora_for_giga_world_1.py)

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

---

## 7. 📁 Repository Layout

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

## 🙏 Acknowledgements

GigaWorld-1 stands on the shoulders of a vibrant open-source ecosystem. We are deeply grateful to the following communities and projects that made this work possible:

### 🤗 Foundation Models & Architectures
- [**Wan (Alibaba)**](https://github.com/Wan-Video/Wan2.1) — the `wan2.1` and `wan2.2_5b` backbones that power GigaWorld-1 Nano and Pro
- [**Diffusers**](https://github.com/huggingface/diffusers) — the modular diffusion framework we extend with custom attention processors and pipelines
- [**Helios**](https://github.com/PKU-YuanGroup/Helios) — a video generation model that achieves minute-scale, high-quality video synthesis
- [**Genesis**](https://github.com/xiaomi-research/genesis) — a generative universal physics engine and robotics/embodied AI simulation platform
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

---

<div align="center">

<sub>Built with ❤️ by the <b>GigaWorld Team, GigaAI</b> · CVPR 2026</sub>

<sub>Released under the Apache 2.0 License.</sub>

</div>
