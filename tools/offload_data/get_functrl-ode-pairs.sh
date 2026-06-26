#!/bin/bash
set -e

DEBUG=${DEBUG:-0}

export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"

if [ "$DEBUG" -eq 1 ]; then
    unset TORCH_LOGS
    unset TORCHDYNAMO_VERBOSE
    export TORCH_NCCL_ENABLE_MONITORING=0

    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
    export NCCL_SHM_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_PXN_DISABLE=1
    unset NCCL_IB_GID_INDEX
    unset NCCL_IB_HCA
    unset NCCL_SOCKET_IFNAME

    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

    MASTER_ADDR="localhost"
    MASTER_PORT=${MASTER_PORT:-12346}
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=1

    echo -e "\033[32m🐛 ======== Single-card Debug Mode ======== 🐛\033[0m"
else
    export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
    export TORCHDYNAMO_VERBOSE=1
    export TORCH_NCCL_ENABLE_MONITORING=1

    export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
    export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE:-}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
    export NCCL_SOCKET_TIMEOUT=${NCCL_SOCKET_TIMEOUT:-3600000}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
    export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
    export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
    export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-0}
    export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
    export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
    export NCCL_IB_TC=${NCCL_IB_TC:-160}
    export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}

    MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-localhost}

    if [ -n "${METIS_WORKER_0_PORT:-}" ]; then
        ports=($(echo "$METIS_WORKER_0_PORT" | tr ',' ' '))
        MASTER_PORT=${ports[0]}
    else
        MASTER_PORT=${MASTER_PORT:-12346}
    fi

    NNODES=${ARNOLD_WORKER_NUM:-1}
    NODE_RANK=${ARNOLD_ID:-0}
    if [ -n "${GPUS_PER_NODE:-}" ]; then
        GPUS_PER_NODE=$GPUS_PER_NODE
    elif [ -n "${ARNOLD_WORKER_GPU:-}" ]; then
        GPUS_PER_NODE=$ARNOLD_WORKER_GPU
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    else
        GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
    fi

    echo -e "\033[31m🚀 ======== Multi-node / Fallback Mode ======== 🚀\033[0m"
fi

WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

if [ -n "${RDZV_BACKEND:-}" ]; then
    DISTRIBUTED_ARGS+=(--rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" --rdzv_id 9863 --rdzv_backend c10d)
    export NCCL_SHM_DISABLE=1
fi

echo -e "\033[31m⚙️  DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS[*]}\033[0m"

BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/mnt/pfs/users/zhanqian.wu/ckpt/stage-1/stage1_final"}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-"/mnt/pfs/users/zhanqian.wu/ckpt/stage-3-init/stage1_final_3v_uni_s16k"}
LORA_PATH=${LORA_PATH:-""}
PARTIAL_PATH=${PARTIAL_PATH:-""}

HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-1920}
NUM_FRAMES=${NUM_FRAMES:-99}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
LATENT_WINDOW_SIZE=${LATENT_WINDOW_SIZE:-9}
WEIGHT_DTYPE=${WEIGHT_DTYPE:-"bf16"}
VAE_DECODE_TYPE=${VAE_DECODE_TYPE:-"default"}
TIME_SHIFT_TYPE=${TIME_SHIFT_TYPE:-"linear"}
ZERO_STEPS=${ZERO_STEPS:-1}
SEED=${SEED:-42}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-25}

FEATURE_FOLDER=${FEATURE_FOLDER:-""}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-""}

echo ""
echo -e "\033[36m╔══════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[36m║         🎥 FunCtrl ODE Pairs - Configuration        ║\033[0m"
echo -e "\033[36m╠══════════════════════════════════════════════════════╣\033[0m"
echo -e "\033[36m║\033[0m  🏷️  Mode             : $( [ "$DEBUG" -eq 1 ] && echo '🐛 Single-card Debug' || echo '🚀 Multi-node / Fallback' )"
echo -e "\033[36m║\033[0m  🌐 WORLD_SIZE       : $WORLD_SIZE  \(NNODES=$NNODES, GPUS_PER_NODE=$GPUS_PER_NODE\)"
echo -e "\033[36m║\033[0m  🖥️  MASTER_ADDR      : $MASTER_ADDR"
echo -e "\033[36m║\033[0m  🔌 MASTER_PORT       : $MASTER_PORT"
echo -e "\033[36m║\033[0m  📋 NODE_RANK         : $NODE_RANK"
echo -e "\033[36m╠══════════════════════════════════════════════════════╣\033[0m"
echo -e "\033[36m║\033[0m  📦 BASE_MODEL_PATH   : $BASE_MODEL_PATH"
echo -e "\033[36m║\033[0m  📦 TRANSFORMER_PATH  : $TRANSFORMER_PATH"
echo -e "\033[36m║\033[0m  🎨 LORA_PATH         : ${LORA_PATH:-'(none)'}"
echo -e "\033[36m║\033[0m  🧩 PARTIAL_PATH      : ${PARTIAL_PATH:-'(none)'}"
echo -e "\033[36m║\033[0m  📐 Height            : $HEIGHT"
echo -e "\033[36m║\033[0m  📐 Width             : $WIDTH"
echo -e "\033[36m║\033[0m  🎞️  Num Frames        : $NUM_FRAMES"
echo -e "\033[36m║\033[0m  🎚️  Guidance Scale    : $GUIDANCE_SCALE"
echo -e "\033[36m║\033[0m  🪟  Latent Window     : $LATENT_WINDOW_SIZE"
echo -e "\033[36m║\033[0m  🔢 Weight Dtype      : $WEIGHT_DTYPE"
echo -e "\033[36m║\033[0m  🎬 VAE Decode Type   : $VAE_DECODE_TYPE"
echo -e "\033[36m║\033[0m  ⏱️  Time Shift Type   : $TIME_SHIFT_TYPE"
echo -e "\033[36m║\033[0m  🎲 Seed              : $SEED"
echo -e "\033[36m║\033[0m  🔢 Num Inference Steps : $NUM_INFERENCE_STEPS"
echo -e "\033[36m║\033[0m  📂 FEATURE_FOLDER    : ${FEATURE_FOLDER:-'(use default in script)'}"
echo -e "\033[36m║\033[0m  📂 OUTPUT_FOLDER     : ${OUTPUT_FOLDER:-'(use default in script)'}"
echo -e "\033[36m╚══════════════════════════════════════════════════════╝\033[0m"
echo ""

LORA_ARG=()
[ -n "$LORA_PATH" ] && LORA_ARG=(--lora_path "$LORA_PATH")

PARTIAL_ARG=()
[ -n "$PARTIAL_PATH" ] && PARTIAL_ARG=(--partial_path "$PARTIAL_PATH")

FEATURE_ARG=()
[ -n "$FEATURE_FOLDER" ] && FEATURE_ARG=(--feature_folder "$FEATURE_FOLDER")

OUTPUT_ARG=()
[ -n "$OUTPUT_FOLDER" ] && OUTPUT_ARG=(--output_folder "$OUTPUT_FOLDER")

MAX_RETRIES=${MAX_RETRIES:-999}
RETRY_INTERVAL=${RETRY_INTERVAL:-60}

retry_count=0

while true; do
    echo ""
    echo "🚀 Launch attempt $((retry_count+1))"
    echo ""

    torchrun "${DISTRIBUTED_ARGS[@]}" \
        tools/offload_data/get_functrl-ode-pairs.py \
        --base_model_path "$BASE_MODEL_PATH" \
        --transformer_path "$TRANSFORMER_PATH" \
        "${LORA_ARG[@]}" \
        "${PARTIAL_ARG[@]}" \
        "${FEATURE_ARG[@]}" \
        "${OUTPUT_ARG[@]}" \
        --use_dynamic_shifting \
        --time_shift_type "$TIME_SHIFT_TYPE" \
        --num_frames "$NUM_FRAMES" \
        --num_inference_steps "$NUM_INFERENCE_STEPS" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --latent_window_size "$LATENT_WINDOW_SIZE" \
        --weight_dtype "$WEIGHT_DTYPE" \
        --vae_decode_type "$VAE_DECODE_TYPE" \
        --seed "$SEED" \
        --use_cfg_zero_star \
        --use_zero_init \
        --zero_steps "$ZERO_STEPS"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "🎉 Job finished successfully."
        exit 0
    fi

    retry_count=$((retry_count + 1))

    echo ""
    echo "❌ Job failed. Exit code: $exit_code"
    echo "🔁 Retry: $retry_count / $MAX_RETRIES"
    echo ""

    if [ $retry_count -ge $MAX_RETRIES ]; then
        echo "💥 Reached retry limit."
        exit $exit_code
    fi

    echo "⏳ Sleep ${RETRY_INTERVAL}s before restart..."
    sleep $RETRY_INTERVAL

    pkill -9 -f get_functrl-ode-pairs.py || true
    pkill -9 -f torchrun || true

    sleep 5
done