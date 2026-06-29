#!/bin/bash
set -e

############################################################
# Environment
############################################################
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR

############################################################
# Torch
############################################################
export TOKENIZERS_PARALLELISM=false
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"

############################################################
# Single-node multi-GPU
############################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=127.0.0.1
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
IFS=',' read -ra GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
GPUS_PER_NODE=${#GPU_ARR[@]}

DISTRIBUTED_ARGS="\
--nproc_per_node ${GPUS_PER_NODE} \
--nnodes ${NNODES} \
--node_rank ${NODE_RANK} \
--master_addr ${MASTER_ADDR} \
--master_port ${MASTER_PORT}
"

############################################################
# Auto-restart settings
############################################################
MAX_RESTARTS=999999
RESTART_INTERVAL=10

############################################################
# Logging
############################################################
LOG_DIR=./logs
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_FILE=${LOG_DIR}/latent_preprocess_${TIMESTAMP}.log

echo -e "\033[31mDISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}\033[0m"
echo -e "\033[32mCUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}\033[0m"
echo -e "\033[34mLOG_FILE=${LOG_FILE}\033[0m"

############################################################
# Infinite auto-restart loop
############################################################
RESTART_COUNT=0

while true
do
    echo ""
    echo "======================================================="
    echo "🚀 START JOB"
    echo "🔁 Restart Count: ${RESTART_COUNT}"
    echo "🕒 Time: $(date)"
    echo "======================================================="
    echo ""

    torchrun ${DISTRIBUTED_ARGS} \
        tools/offload_data/get_short-latents-giga-ctrl.py \
        --append_mode \
        --order_mode random \
        2>&1 | tee -a ${LOG_FILE}

    EXIT_CODE=${PIPESTATUS[0]}

    echo ""
    echo "======================================================="
    echo "⚠️ PROCESS EXITED"
    echo "❌ EXIT_CODE=${EXIT_CODE}"
    echo "🕒 Time: $(date)"
    echo "======================================================="
    echo ""

    ########################################################
    # Normal exit
    ########################################################
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "🎉 Job finished successfully."
        break
    fi

    ########################################################
    # Restart limit
    ########################################################
    RESTART_COUNT=$((RESTART_COUNT + 1))

    if [ ${RESTART_COUNT} -ge ${MAX_RESTARTS} ]; then
        echo "❌ Reached MAX_RESTARTS=${MAX_RESTARTS}"
        exit 1
    fi

    ########################################################
    # Wait before restart
    ########################################################
    echo "⏳ Restart after ${RESTART_INTERVAL}s..."
    sleep ${RESTART_INTERVAL}
done