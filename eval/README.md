# <u>Evaluation Pipeline</u> by *Gigaworld*
This repository shows how to evaluate custom models described in the [Gigaworld](https://arxiv.org/abs/2603.04379) paper.

## 🎉 Overview

### Basic Metrics
Measuring the basic quality of videos.

| Metric | Description | Method |
|--------|-------------|-------|
| **Aesthetic** | Aesthetic quality score | CLIP + LAION Aesthetic |
| **Motion Amplitude** | Motion dynamics degree | Farneback |
| **Motion Smoothness** | Temporal motion smoothness | AMT |
| **Semantic** | Overall semantic consistency | ViCLIP |
| **Naturalness** | Overall semantic consistency | GPT-5.2-2025-12-11 |

### Drifting Metrics
Measuring start-end quality contrast to detect temporal drifting:

$$\Delta M_{drift}(V) = |M(V_{start}) - M(V_{end})|$$

Where $V_{start}$ is the first 15% of frames and $V_{end}$ is the last 15% of frames.

| Metric | Description |
|--------|-------------|
| **Drifting Aesthetic** | Aesthetic drifting |
| **Drifting Motion Smoothness** | Motion smoothness drifting |
| **Drifting Semantic** | Semantic consistency drifting |
| **Drifting Naturalness** | Naturalness drifting |

### Throughput Metrics
Measureing the end-to-end performance at a resolution of 384 × 640 under default frame lengths. The reported results include the latency of both the VAE and the text encoder. We should enable all acceleration techniques officially adopted by each model—such as FlashAttention, torch compile,
KV-cache, and warm-up—to achieve optimal throughput.

| Metric | Description |
|--------|-------------|
| **Throughput (FPS)** | Inference speed |
| **Throughput Score** | Inference speed |


## ⚙️ Requirements and Installation

### Prepare Environment

```bash
# Activate conda environment
conda activate gigaworld

# Install additional dependencies
pip install -r requirements.txt
```

### Prepare Ckpts
```bash
# Option 1: via script
cd checkpoints/ && bash get_checkpoints.sh

# Option 2: via Hugging Face CLI
hf download BestWishYsh/GigaworldBench-Weights --local-dir ./
```

Once ready, the weights will be organized in this format:

```
📦 checkpoints/
├── 📂 aesthetic_model/
│   ├── 📄 sa_0_4_vit_l_14_linear.pth
│   └── 📄 ViT-L-14.pt
└── 📂 ViCLIP/
    ├── 📄 bpe_simple_vocab_16e6.txt.gz
    └── 📄 ViClip-InternVid-10M-FLT.pth
```


## 🗝️ Usage

### Prepare Your Videos

```
📦 model_name/
├── 📄 1_*_ori*.mp4
├── 📄 2_*_ori*.mp4
├── ...
└── 📄 {id}_{target-duration}_{true-duration}.mp4
```

### Run Metrics:

```bash
# Option 1: Run all metrics (recommended)
bash run_metrics.sh          # for single GPU
bash run_metrics_ddp.sh      # for multi-GPU

# Option 2: Run individual scripts (same results)
python 0_get_aesthetic.py
python 1_get_motion_amplitude.py
python 2_get_motion_smoothness.py
python 3_get_semantic.py
python 4_get_naturalness.py

python 5_get_drifting_aesthetic.py
python 6_get_drifting_motion_smoothness.py
python 7_get_drifting_semantic.py
python 8_get_drifting_naturalness.py
```

## Output

Results are saved as JSON files in the `playground/results` directory:

```bash
# Convert raw metrics to rating and merge them into one json
python 9_merge_all_scores.py
python 10_merge_all_results.py
```

A merged summary is saved to `playground/results/merged_results.json`.

```json
{
  "num_models": 1,
  "score_type": "rating",
  "metrics": [
    "aesthetic",
    "drifting_aesthetic",
    "drifting_motion_smoothness",
    "drifting_naturalness",
    "drifting_semantic",
    "motion_amplitude",
    "motion_smoothness",
    "naturalness",
    "semantic",
    "total_weighted_rating"
  ],
  "models": {
    "toy-video": {
      "aesthetic": 9,
      "motion_amplitude": 3,
      "motion_smoothness": 10,
      "naturalness": 7,
      "semantic": 8,
      "drifting_aesthetic": 8,
      "drifting_motion_smoothness": 10,
      "drifting_naturalness": 10,
      "drifting_semantic": 10,
      "total_weighted_rating": 8.247
    }
  },
  "rating_scale": 10
}
```

For the **Throughput Score**, you should first measure the end-to-end throughput (in FPS). The score increases by 1 point for every 3.2 FPS and is clipped to the range $[1, 10]$ like other metrics. Formally,

$$
\text{Throughput Score} =
\begin{cases}
1, & \text{if } \text{FPS} \le 3.2, \\
\left\lceil \dfrac{\text{FPS}}{3.2} \right\rceil, & \text{if } 3.2 < \text{FPS} < 32, \\
10, & \text{if } \text{FPS} \ge 32.
\end{cases}
$$

## 🔒 Acknowledgement

* This project wouldn't be possible without the following open-sourced repositories: [OpenS2V-Nexus](https://github.com/PKU-YuanGroup/OpenS2V-Nexus), [VBench](https://github.com/Vchitect/VBench), [ChronoMagic-Bench](https://github.com/PKU-YuanGroup/ChronoMagic-Bench), [FramePack](https://github.com/lllyasviel/FramePack).
* Existing metrics are insufficient for accurately assessing the performance of video generation models. A promising direction is to develop perceptually aligned metrics that better reflect human judgment.
