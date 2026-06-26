# <u>Data Preprocessing Pipeline</u> by *Helios*
This repository describes the data preprocessing pipeline used in the [Helios](https://arxiv.org/abs/2603.04379) paper. And we prepare a toy training data [here](https://huggingface.co/BestWishYsh/HeliosBench-Weights/tree/main/demo_data).


## ⚙️ Requirements and Installation


### Environment

```bash
# Activate conda environment
conda activate helios
```

## 🗝️ Usage

### Step 1 - Prepare Metadata and Organize Videos

To train your own video generation model, create JSON files following this [format](./example/toy_data/toy_filter.json):

```
[
    {
        "cut": [0, 81],
        "crop": [0, 832, 0, 480],
        "fps": 24.0,
        "num_frames": 81,
        "resolution": {
        "height": 480,
        "width": 832
        },
        "cap": [
        "A stunning mid-afternoon ..."
        ],
        "path": "videos/2_240_ori81.mp4"
    },
    {
        "cut": [0, 81],
        ...
    }
...
]
```

and arrange video files following this [structure](./example):

```
📦 example/
├── 📂 toy_data/
│   ├── 📂 videos
│   │   ├── 2_240_ori81.mp4
│   │   ├── 239_120_ori129.mp4.mp4
│   │   └── ...
│   └── 📄 toy_data_1.json
│
├── 📂 toy_data_2/
│   │   ├── A.mp4
│   │   ├── B.mp4
│   │   └── ...
│   └── 📄 toy_data_2.json
...
```

### Step 2 - Prepare Autoregressive Real Data

These data can be used for training Stage-1, Stage-2, and Stage-3.

```bash
# Remember to modify the input and output paths before running
bash get_short-latents.py
```

### Step 3 - Prepare Autoregressive ODE Data

These data can only be used for training Stage-3.

```bash
# Remember to modify the input and output paths before running
bash get_ode-pairs.sh
```

### (Optional) Step 4 - Prepare Text Data 

If you want to use the [Self-Forcing](https://github.com/guandeh17/Self-Forcing) training approach, prepare text embeddings:

```bash
# Remember to modify the input and output paths before running
bash get_text-embedding.sh
```

## 🔒 Acknowledgement

* This project wouldn't be possible without the following open-sourced repositories: [OpenSora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [OpenSora](https://github.com/hpcaitech/Open-Sora), [Video-Dataset-Scripts](https://github.com/huggingface/video-dataset-scripts)