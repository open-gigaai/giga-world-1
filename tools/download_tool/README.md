# Giga-World-1 Download Tool

This folder provides a one-click download tool for Giga-World-1 model weights and toy data.

Supported sources:

- Hugging Face model: `https://huggingface.co/GigaAI-Research/Giga-World-1`
- Hugging Face toy data: `https://huggingface.co/datasets/GigaAI-Research/Giga-World-1-Toydata`
- ModelScope model: `https://modelscope.cn/models/GigaAI/Giga-World-1/summary`
- ModelScope toy data: `https://modelscope.cn/datasets/GigaAI/Giga-World-1-Toydata`

## Files

```text
tools/download_tool/
├── download_giga_world.py   # Python downloader
├── download_giga_world.sh   # Bash wrapper
└── README.md                # Usage guide
```

## Installation

For Hugging Face downloads:

```bash
pip install huggingface_hub
```

For ModelScope downloads:

```bash
pip install modelscope
```

If you use Git-based download, install Git LFS first:

```bash
git lfs install
```

## Basic Usage

Run from the repository root:

```bash
cd giga-world-release
```

### Download from Hugging Face

Download model weights:

```bash
python tools/download_tool/download_giga_world.py \
  --platform hf \
  --target model \
  --output-dir ./downloads
```

Download toy data:

```bash
python tools/download_tool/download_giga_world.py \
  --platform hf \
  --target toydata \
  --output-dir ./downloads
```

Download both model weights and toy data:

```bash
python tools/download_tool/download_giga_world.py \
  --platform hf \
  --target all \
  --output-dir ./downloads
```

### Download from ModelScope

Download model weights:

```bash
python tools/download_tool/download_giga_world.py \
  --platform modelscope \
  --target model \
  --output-dir ./downloads
```

Download toy data:

```bash
python tools/download_tool/download_giga_world.py \
  --platform modelscope \
  --target toydata \
  --output-dir ./downloads
```

Download both model weights and toy data:

```bash
python tools/download_tool/download_giga_world.py \
  --platform modelscope \
  --target all \
  --output-dir ./downloads
```

## Download Method

The default method is SDK download:

```bash
python tools/download_tool/download_giga_world.py \
  --platform hf \
  --target model \
  --method sdk \
  --output-dir ./downloads
```

You can also use Git download:

```bash
python tools/download_tool/download_giga_world.py \
  --platform hf \
  --target model \
  --method git \
  --output-dir ./downloads
```

## Authentication

If the repository requires authentication, pass a token explicitly:

```bash
python tools/download_tool/download_giga_world.py \
  --platform hf \
  --target model \
  --output-dir ./downloads \
  --token YOUR_HF_TOKEN
```

Or set environment variables:

```bash
export HF_TOKEN=YOUR_HF_TOKEN
export MODELSCOPE_TOKEN=YOUR_MODELSCOPE_TOKEN
```

## Proxy Example

If needed, configure proxy before downloading:

```bash
unset all_proxy
unset ALL_PROXY

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
```

## Output Structure

The downloaded folders will be saved under `--output-dir`:

```text
./downloads/
├── Giga-World-1/              # model weights
└── Giga-World-1-Toydata/      # toy data
```

## Bash Wrapper

You can also use the shell wrapper:

```bash
bash tools/download_tool/download_giga_world.sh \
  --platform hf \
  --target all \
  --output-dir ./downloads
```

If you want to run it directly:

```bash
chmod +x tools/download_tool/download_giga_world.sh

./tools/download_tool/download_giga_world.sh \
  --platform hf \
  --target all \
  --output-dir ./downloads
```
