export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="online"
export WANDB_API_KEY="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"
export NCCL_TIMEOUT="360000000"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TIMEOUT="360000000"

NUM_MACHINES=1
MACHINE_RANK=0
NUM_PROCESSES=1

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
accelerate launch \
    ${ACCELERATE_ARGS} \
    --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
    train_gigaworld_functrl_uni_stage1.py \
    --config scripts/training/configs/stage_1_post_functrl_wan21.yaml \
    2>&1 | tee ./train.log
