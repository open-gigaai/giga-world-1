#!/bin/bash
# ============================================================
# gigatrans2helios 启动脚本 (支持自动重启)
# 用法: bash run_gigatrans2helios.sh
# ============================================================

set -euo pipefail

# ----- GPU 配置 -----
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

# ----- 运行参数 -----
# NUM_WORKERS_PER_GPU 自动从 CUDA_VISIBLE_DEVICES 推导 (见 Python 脚本)
export DA2_BATCH_SIZE=2
export DA2_MODEL_SIZE="large"

# ----- 视频参数 -----
export NUM_FRAMES=121
export STRIDE=90
export FPS=10

# ----- 增量/重置 -----
export RESET_JSONL="0"       # 1=清空重跑 jsonl
export RESET_VIDEOS="0"      # 1=重写视频文件
export ENABLE_APPEND_MODE="1" # 1=增量追加

# ----- 自动重启配置 -----
MAX_RETRIES=10               # 最大重试次数 (0=无限重试)
RETRY_DELAY=30               # 重启前等待秒数
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/gigatrans2helios.py"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ============================================================

attempt=0

while true; do
    attempt=$((attempt + 1))
    timestamp=$(date '+%Y%m%d_%H%M%S')
    log_file="${LOG_DIR}/run_${timestamp}_attempt${attempt}.log"

    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt ${attempt} starting..."
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "  Log: ${log_file}"
    echo "=========================================="

    python -u "${PYTHON_SCRIPT}" 2>&1 | tee "${log_file}"
    exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Completed successfully."
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed with exit code ${exit_code}."

    if [ ${MAX_RETRIES} -gt 0 ] && [ ${attempt} -ge ${MAX_RETRIES} ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⛔ Reached max retries (${MAX_RETRIES}). Giving up."
        exit ${exit_code}
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Restarting in ${RETRY_DELAY}s..."
    sleep ${RETRY_DELAY}
done
