#!/bin/bash
set -e

#################################################################
## Wandb (set WANDB_API_KEY in your environment to enable online mode)
#################################################################
export WANDB_MODE=${WANDB_MODE:-"offline"}
# export WANDB_API_KEY=""

#################################################################
## NCCL
#################################################################
export NCCL_TIMEOUT="360000000"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TIMEOUT="360000000"

#################################################################
## Single-node multi-GPU (auto-detect all visible GPUs)
#################################################################
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${NUM_GPUS}" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    else
        echo "No GPUs detected via nvidia-smi; defaulting to CUDA_VISIBLE_DEVICES=0"
        export CUDA_VISIBLE_DEVICES=0
    fi
fi

NUM_MACHINES=1
MACHINE_RANK=0
NUM_PROCESSES=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

ACCELERATE_ARGS="\
--num_machines ${NUM_MACHINES} \
--machine_rank ${MACHINE_RANK} \
--num_processes ${NUM_PROCESSES} \
"

echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"
echo -e "\033[32mCUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}\033[0m"

#################################################################
## Launch
#################################################################
OUTPUT_DIR=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('scripts/training/configs/stage_2_dmd_functrl_wan22_5b.yaml')['output_dir'])")
mkdir -p ${OUTPUT_DIR}

accelerate launch \
    ${ACCELERATE_ARGS} \
    --config_file scripts/accelerate_configs/example_zero2.yaml \
    train_gigaworld_functrl_uni_stage2_dmd.py \
    --config scripts/training/configs/stage_2_dmd_functrl_wan22_5b.yaml \
    2>&1 | tee ${OUTPUT_DIR}/train.log