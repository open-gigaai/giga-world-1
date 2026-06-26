# #!/bin/bash
# export WANDB_MODE="offline"
# export WANDB_API_KEY=""
# export TOKENIZERS_PARALLELISM=true

# export OMNISTORE_LOAD_STRICT_MODE=0
# export OMNISTORE_LOGGING_LEVEL=ERROR
# #################################################################
# ## Torch
# #################################################################
# export TOKENIZERS_PARALLELISM=false
# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_NCCL_ENABLE_MONITORING=1
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
# #################################################################


# #################################################################
# ## NCCL
# #################################################################
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_SOCKET_TIMEOUT=3600000

# export NCCL_DEBUG=WARN  # disable the verbose NCCL logs
# export NCCL_P2P_DISABLE=0
# export NCCL_IB_DISABLE=0  # was 1
# export NCCL_SHM_DISABLE=0  # was 1
# export NCCL_P2P_LEVEL=NVL

# export NCCL_PXN_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_TC=160
# export NCCL_IB_TIMEOUT=22
# #################################################################

# # #################################################################
# # ## DIST
# # #################################################################
# # MASTER_ADDR=$ARNOLD_WORKER_0_HOST
# # ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# # export MASTER_PORT=${ports[0]}
# # NNODES=$ARNOLD_WORKER_NUM
# # NODE_RANK=$ARNOLD_ID
# # GPUS_PER_NODE=$ARNOLD_WORKER_GPU
# # # GPUS_PER_NODE=1
# # # NNODES=1
# # # NODE_RANK=0
# # WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# # DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# # if [ ! -z $RDZV_BACKEND ]; then
# #     DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_id 9863 --rdzv_backend c10d"
# #     export NCCL_SHM_DISABLE=1
# # fi

# # echo -e "\033[31mDISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}\033[0m"

# #################################################################
# ## ACCELERATE CONFIG
# #################################################################
# MASTER_ADDR=$ARNOLD_WORKER_0_HOST
# ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# export MASTER_PORT=${ports[0]}
# NUM_MACHINES=$ARNOLD_WORKER_NUM
# MACHINE_RANK=$ARNOLD_ID
# NUM_PROCESSES_PER_MACHINE=$ARNOLD_WORKER_GPU

# export CUDA_VISIBLE_DEVICES=0
# NUM_PROCESSES_PER_MACHINE=1
# NUM_MACHINES=1
# MACHINE_RANK=0

# ACCELERATE_ARGS="--num_machines $NUM_MACHINES --machine_rank $MACHINE_RANK --num_processes $((NUM_PROCESSES_PER_MACHINE*NUM_MACHINES)) --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"

# echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"

# # accelerate launch \
# #     $ACCELERATE_ARGS \
# #     train_gigaworld.py \
# #     --config scripts/training/configs/stage_2_post.yaml \
# #     2>&1 | tee ./train.log

# accelerate launch \
#     $ACCELERATE_ARGS \
#     --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
#     train_gigaworld.py \
#     --config scripts/training/configs/stage_1_init.yaml \
#     2>&1 | tee ./train.log

# ####################################################################
# #!/bin/bash
# set -e

# export WANDB_MODE="offline"
# export WANDB_API_KEY=""
# export TOKENIZERS_PARALLELISM=false

# export OMNISTORE_LOAD_STRICT_MODE=0
# export OMNISTORE_LOGGING_LEVEL=ERROR

# #################################################################
# ## Torch
# #################################################################
# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_NCCL_ENABLE_MONITORING=0
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"

# #################################################################
# ## 单机单卡
# #################################################################
# export CUDA_VISIBLE_DEVICES=0

# NUM_MACHINES=1
# MACHINE_RANK=0
# NUM_PROCESSES=1

# ACCELERATE_ARGS="\
# --num_machines ${NUM_MACHINES} \
# --machine_rank ${MACHINE_RANK} \
# --num_processes ${NUM_PROCESSES}
# "

# echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"
# echo -e "\033[32mCUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}\033[0m"

# #################################################################
# ## Launch
# #################################################################
# accelerate launch \
#     ${ACCELERATE_ARGS} \
#     --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
#     train_gigaworld_ctrl_stage1.py \
#     --config scripts/training/configs/stage_1_post_ctrl.yaml \
#     2>&1 | tee ./train.log

# accelerate launch \
#     ${ACCELERATE_ARGS} \
#     --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
#     train_gigaworld.py \
#     --config scripts/training/configs/stage_1_init.yaml \
#     2>&1 | tee ./train.log

# set -e

# export WANDB_MODE="online"
# export WANDB_API_KEY=""
# export TOKENIZERS_PARALLELISM=false

# export OMNISTORE_LOAD_STRICT_MODE=0
# export OMNISTORE_LOGGING_LEVEL=ERROR

# #################################################################
# ## Torch
# #################################################################
# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_NCCL_ENABLE_MONITORING=0
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"

#################################################################
## 分布式配置
#################################################################
# RUN_MODE=single_node: 单机多卡
# RUN_MODE=volcano:     火山多机多卡
# 默认单机，避免本地无 MLP_* 环境变量时参数为空
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
    --config scripts/training/configs/stage_2_dmd_functrl_wan22_5b.yaml\
    2>&1 | tee ./train.log
