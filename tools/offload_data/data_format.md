# Data Format

This document describes the input and output dataset formats used by the data-preprocessing scripts:

- `get_short-latents-giga-ctrl.py`
- `get_short-latents-giga-ctrl-wan22-5b.py`

Both scripts share the same data interface; they differ only in the pretrained model and default output directory.

## 1. Input Dataset

### 1.1 Directory layout

```text
<data_root>/
├── helios_giga_ctrl.jsonl   # metadata
├── videos/                  # RGB videos
└── control_videos/          # control videos (frame-aligned with RGB)
```

### 1.2 Metadata (`*.jsonl`)

One JSON object per line. Each sample must contain at least:

| field | type | description |
|---|---|---|
| `uttid` | str | unique sample ID |
| `bucket_key` | str | bucket id used by the sampler |
| `bucket_num_frame` | int | number of frames |
| `bucket_height` | int | frame height |
| `bucket_width` | int | frame width |

Example:

```json
{"uttid":"sample_000001","bucket_key":"480x1920x49","bucket_num_frame":49,"bucket_height":480,"bucket_width":1920}
```

> Note: the full schema is defined inside `BucketedFeatureDataset`. The fields above are the minimum required by these scripts.

### 1.3 Batch fields (produced by the dataloader)

| key | shape | notes |
|---|---|---|
| `uttid` | list[str] | sample IDs |
| `video_metadata` | dict | lists of `num_frames` / `height` / `width` |
| `videos` | `[B,T,C,H,W]` | RGB video; permuted to `[B,C,T,H,W]` for the VAE |
| `control_videos` | `[B,T,C,H,W]` | control video; same handling |
| `prompts` | list[str] | text prompts |
| `first_frames_images` | `[B,C,H,W]` | first frame, saved as PIL |

## 2. Output Dataset

### 2.1 Directory layout

```text
<output_root>/
├── {uttid}_{num_frame}_{height}_{width}.pt
├── sample_000001_49_480_1920.pt
├── sample_000002_49_480_1920.pt
└── ...
```

- one `.pt` file per sample, named `{uttid}_{num_frame}_{height}_{width}.pt`
- a file is considered valid when its size ≥ 1024 bytes (used by append mode to skip)

### 2.2 `.pt` contents

A dictionary saved via `torch.save`:

| key | type | description |
|---|---|---|
| `vae_latent` | Tensor (CPU) | RGB video latent |
| `control_latent` | Tensor (CPU) | control video latent |
| `prompt_embed` | Tensor (CPU) | text embedding |
| `prompt_attention_mask` | Tensor (CPU) | attention mask |
| `first_frames_image` | PIL.Image | first frame |
| `prompt_raw` | str | original prompt |
| `control_type` | str | constant `"external_control_video"` |
