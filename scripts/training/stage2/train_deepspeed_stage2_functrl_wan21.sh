RUN_MODE=${RUN_MODE:-single_node}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="online"
export WANDB_API_KEY="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"
export NCCL_TIMEOUT="360000000"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TIMEOUT="360000000"

if [ "${RUN_MODE}" = "single_node" ]; then
    #################################################################
    ## 单机多卡
    #################################################################
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

    NUM_MACHINES=1
    MACHINE_RANK=0
    NUM_PROCESSES=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

    ACCELERATE_ARGS="\
--num_machines ${NUM_MACHINES} \
--machine_rank ${MACHINE_RANK} \
--num_processes ${NUM_PROCESSES} \
"

elif [ "${RUN_MODE}" = "volcano" ]; then
    #################################################################
    ## 火山多机多卡
    #################################################################
    # 火山环境变量：
    # MLP_WORKER_GPU      当前实例 GPU 数 / 每节点进程数
    # MLP_WORKER_NUM      实例数 / 节点数
    # MLP_ROLE_INDEX      当前节点 rank
    # MLP_WORKER_0_HOST   主节点地址
    # MLP_WORKER_0_PORT   主节点端口

    : "${MLP_WORKER_GPU:?MLP_WORKER_GPU is required in volcano mode}"
    : "${MLP_WORKER_NUM:?MLP_WORKER_NUM is required in volcano mode}"
    : "${MLP_ROLE_INDEX:?MLP_ROLE_INDEX is required in volcano mode}"
    : "${MLP_WORKER_0_HOST:?MLP_WORKER_0_HOST is required in volcano mode}"
    : "${MLP_WORKER_0_PORT:?MLP_WORKER_0_PORT is required in volcano mode}"

    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((MLP_WORKER_GPU - 1)))

    NUM_MACHINES=${MLP_WORKER_NUM}
    MACHINE_RANK=${MLP_ROLE_INDEX}
    NUM_PROCESSES=$((MLP_WORKER_GPU * MLP_WORKER_NUM))
    MASTER_ADDR=${MLP_WORKER_0_HOST}
    MASTER_PORT=${MLP_WORKER_0_PORT}

    ACCELERATE_ARGS="\
--num_machines ${NUM_MACHINES} \
--machine_rank ${MACHINE_RANK} \
--num_processes ${NUM_PROCESSES} \
--main_process_ip ${MASTER_ADDR} \
--main_process_port ${MASTER_PORT} \
"
else
    echo "Unknown RUN_MODE=${RUN_MODE}, expected single_node or volcano"
    exit 1
fi

echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"
echo -e "\033[32mCUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}\033[0m"

#################################################################
## Launch
#################################################################
accelerate launch \
    ${ACCELERATE_ARGS} \
    --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
    train_gigaworld_functrl_uni_stage2_dmd.py\
    --config scripts/training/configs/stage_2_dmd_functrl_wan21.yaml\
    2>&1 | tee ./train.log
