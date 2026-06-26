#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${PROJECT_ROOT}/infer/infer_benchmark.py"

CONFIG_PATH="${PROJECT_ROOT}/scripts/training/configs/stage_1_post_functrl_wan21.yaml"
BASE_MODEL_PATH="/mnt/pfs/users/zhanqian.wu/ckpt/stage-1/stage1_final"
TRANSFORMER_MODEL_PATH="/mnt/pfs/users/zhanqian.wu/ckpt/stage-1/stage1_final"
BASE_OUTPUT_DIR="/mnt/pfs/users/zhanqian.wu/output/giga_eval_rollout"

# 启用的实验组合
# COMBINATIONS=(
#   "checkpoint-1500|ablation_stage_1_post_giga_functrl_lora_task5'$'\342\200\224\342\200\224''0608/checkpoint-1500|/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld/task5"
#   "checkpoint-4000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_0526_task3_overfit/checkpoint-4000|/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld/task3"
#   "checkpoint-1500|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_task1——0608/checkpoint-1500|/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld/task1"
#   "checkpoint-1000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_task2/checkpoint-1000|/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld/task2"
#   "checkpoint-500|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_microwave/checkpoint-500|/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld/microwave"
#   "checkpoint-5500|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_task11/checkpoint-5500|/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld/task11"
#   "checkpoint-3000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_fold_the_shirt_easy/checkpoint-3000|/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld/fold_the_shirt_easy"
#   "checkpoint-2000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_task4/checkpoint-2000|/shared_disk/users/jingyu.liu/own/gw1_data/test_dataset_gigaworld/task4"
# )

COMBINATIONS=(
  "checkpoint-4000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_0526_task3_overfit/checkpoint-4000|/shared_disk/users/zhanqian.wu/data/infer_data/task3_zkey"
)

SEED=42
FPS=10
NUM_FRAMES=429
HEIGHT=480
WIDTH=1920
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=5.0
SAMPLE_NAME="rollout"

export HF_ENABLE_PARALLEL_LOADING=yes
export HF_PARALLEL_LOADING_WORKERS=8
export TOKENIZERS_PARALLELISM=false
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# GPU 设备配置 - 指定哪张卡就用哪张卡（不做任务轮询分配）
CUDA_DEVICES="0"  # 例如 "3" 或 "0,1"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
echo "固定使用 GPU 设备: ${CUDA_VISIBLE_DEVICES}"

cd "${PROJECT_ROOT}"

for i in "${!COMBINATIONS[@]}"; do
  COMBO="${COMBINATIONS[$i]}"
  IFS='|' read -r CHECKPOINT_TAG CHECKPOINT_PATH DATASET_DIR <<< "${COMBO}"
  CAPTIONS_PATH="${DATASET_DIR}/captions.json"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}${DATASET_DIR}"

  echo "============================================================"
  echo "[Run] checkpoint=${CHECKPOINT_PATH}"
  echo "      dataset   =${DATASET_DIR}"
  echo "      output    =${OUTPUT_DIR}"
  echo "============================================================"

  mkdir -p "${OUTPUT_DIR}"

  ARGS=(
    --config "${CONFIG_PATH}"
    --base_model_path "${BASE_MODEL_PATH}"
    --transformer_model_name_or_path "${TRANSFORMER_MODEL_PATH}"
    --dataset_dir "${DATASET_DIR}"
    --captions_path "${CAPTIONS_PATH}"
    --control_video_path "None"
    --image_path "None"
    --prompt "None"
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

  if [[ -n "${CHECKPOINT_PATH}" && "${CHECKPOINT_PATH}" != "None" && "${CHECKPOINT_PATH}" != "none" ]]; then
    ARGS+=(--checkpoint_path "${CHECKPOINT_PATH}")
  fi

  echo "使用固定 GPU: ${CUDA_VISIBLE_DEVICES} 运行任务 $((i+1))/${#COMBINATIONS[@]}"
  python "${SCRIPT_PATH}" "${ARGS[@]}"
done

echo "所有推理任务完成！"
