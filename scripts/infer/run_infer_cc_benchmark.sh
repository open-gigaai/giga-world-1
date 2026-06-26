#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/pfs/users/zhanqian.wu/code/Gigaworld"
SCRIPT_PATH="${PROJECT_ROOT}/infer/infer_cc_benchmark.py"

CONFIG_PATH="${PROJECT_ROOT}/scripts/training/configs/stage_1_post_functrl.yaml"
BASE_MODEL_PATH="/mnt/pfs/users/zhanqian.wu/ckpt/stage-1/stage1_final"
TRANSFORMER_MODEL_PATH="/mnt/pfs/users/zhanqian.wu/ckpt/stage-1/stage1_final"
BASE_OUTPUT_DIR="/mnt/pfs/users/zhanqian.wu/output/giga_eval_rollout"

COMBINATIONS=(
  # "checkpoint-3500|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task1/checkpoint-3500|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task1_rollout_test"
  # "checkpoint-3500|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task1/checkpoint-3500|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task1_test"
  # "checkpoint-2000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task3/checkpoint-2000|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task3_rollout_test"
  # "checkpoint-2000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task3/checkpoint-2000|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task3_test"
  # "checkpoint-2000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task2/checkpoint-2000|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task2_rollout_test"
  # "checkpoint-2000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task2/checkpoint-2000|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task2_test"
  # "checkpoint-1000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task1_small/checkpoint-1000|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task1_rollout_test"
  # "checkpoint-1000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task1_small/checkpoint-1000|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task1_test"
   "checkpoint-1000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task4/checkpoint-1000|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task4_rollout_test"
  "checkpoint-1000|/shared_disk/users/zhanqian.wu/output/experiment/gigaworld/ablation_stage_1_post_giga_functrl_lora_cc_task4/checkpoint-1000|/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl_cc_infer/task4_test"
)

SEED=42
FPS=10
FRAME_MULTIPLE=33
HEIGHT=480
WIDTH=1920
NUM_INFERENCE_STEPS=10
GUIDANCE_SCALE=5.0
SAMPLE_NAME="cc_rollout"
LOAD_LORA=True

export HF_ENABLE_PARALLEL_LOADING=yes
export HF_PARALLEL_LOADING_WORKERS=8
export TOKENIZERS_PARALLELISM=false
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# GPU 设备配置 - 设置可用的 GPU 设备，用于多卡并行推理
CUDA_DEVICES="0,1,2,3"  # 根据实际可用 GPU 修改，例如 "0,1,2,3"
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_DEVICES}"
NUM_GPUS=${#GPU_ARRAY[@]}
echo "使用 GPU 设备: ${CUDA_DEVICES} (共 ${NUM_GPUS} 张)"

cd "${PROJECT_ROOT}"

for i in "${!COMBINATIONS[@]}"; do
  COMBO="${COMBINATIONS[$i]}"
  IFS='|' read -r CHECKPOINT_TAG CHECKPOINT_PATH DATASET_DIR <<< "${COMBO}"
  CAPTIONS_PATH="${DATASET_DIR}/helios_giga_ctrl_cc_infer.jsonl"
  DATASET_NAME="$(basename "${DATASET_DIR}")"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATASET_NAME}"

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
    --frame_multiple "${FRAME_MULTIPLE}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --enable_tiling True
    --load_lora "${LOAD_LORA}"
  )

  if [[ "${LOAD_LORA}" == "True" || "${LOAD_LORA}" == "true" || "${LOAD_LORA}" == "1" ]]; then
    if [[ -n "${CHECKPOINT_PATH}" && "${CHECKPOINT_PATH}" != "None" && "${CHECKPOINT_PATH}" != "none" ]]; then
      ARGS+=(--checkpoint_path "${CHECKPOINT_PATH}")
    fi
  fi

  # 轮询分配 GPU，实现多卡并行
  GPU_ID=${GPU_ARRAY[$((i % NUM_GPUS))]}
  echo "分配 GPU: ${GPU_ID} 给任务 $((i+1))/${#COMBINATIONS[@]}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python "${SCRIPT_PATH}" "${ARGS[@]}" &
  
  # 控制并行度：每启动 NUM_GPUS 个任务后等待完成，避免 GPU 内存溢出
  if (( (i + 1) % NUM_GPUS == 0 )); then
    echo "等待当前批次任务完成..."
    wait
  fi
done

# 等待所有剩余任务完成
wait
echo "所有推理任务完成！"
