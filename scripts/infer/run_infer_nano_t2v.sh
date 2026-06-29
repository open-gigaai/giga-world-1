#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${PROJECT_ROOT}/infer/infer_giga_world.py"

# ---- Model paths ----
CONFIG_PATH="${PROJECT_ROOT}/scripts/training/configs/stage_1_post_functrl_wan21.yaml"
BASE_MODEL_PATH="model/stage1/nano/Giga-World-1-nano-stage1_final-diffusers"
TRANSFORMER_MODEL_PATH="model/stage1/nano/Giga-World-1-nano-stage1_final-diffusers"

# ---- LoRA checkpoint ----
CHECKPOINT_PATH="model/stage1/nano/Giga-World-1-nano-stage1_scene_lora"

# ---- Inputs (t2v: no image_path needed) ----
PROMPT="stack the box ears . The scene features a flat, gray surface with a subtle wood-grain texture, serving as the entire visible background. The only object present is a plain brown cardboard box lying open and centered on this surface. "
CONTROL_VIDEO_PATH="${PROJECT_ROOT}/example/infer_assest/control_video.mp4"

# ---- Output ----
OUTPUT_DIR="${PROJECT_ROOT}/output/infer_results/giga_t2v_nano"
SAMPLE_NAME="t2v_sample"

# ---- Inference params ----
SEED=42
FPS=10
NUM_FRAMES=330
HEIGHT=480
WIDTH=1920
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=5.0

# ---- GPU config ----
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# ---- Environment variables ----
export HF_ENABLE_PARALLEL_LOADING=yes
export HF_PARALLEL_LOADING_WORKERS=8
export TOKENIZERS_PARALLELISM=false
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --config "${CONFIG_PATH}"
  --base_model_path "${BASE_MODEL_PATH}"
  --transformer_model_name_or_path "${TRANSFORMER_MODEL_PATH}"
  --prompt "${PROMPT}"
  --output_dir "${OUTPUT_DIR}"
  --sample_name "${SAMPLE_NAME}"
  --seed "${SEED}"
  --fps "${FPS}"
  --num_frames "${NUM_FRAMES}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --enable_tiling True
)

# Optional: control video
if [[ -n "${CONTROL_VIDEO_PATH}" ]]; then
  ARGS+=(--control_video_path "${CONTROL_VIDEO_PATH}")
fi

# Optional: LoRA checkpoint
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  ARGS+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

python "${SCRIPT_PATH}" "${ARGS[@]}"

echo "Inference done!"
