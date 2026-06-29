#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONDA_SH="/mnt/pfs/users/zhanqian.wu/miniconda3/etc/profile.d/conda.sh"
DEFAULT_ENV_PATH="/mnt/pfs/users/zhanqian.wu/env/Helios"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${DEFAULT_CONDA_SH}" ]]; then
  source "${DEFAULT_CONDA_SH}"
fi

if command -v conda >/dev/null 2>&1; then
  CURRENT_PREFIX="${CONDA_PREFIX:-}"
  if [[ -z "${CURRENT_PREFIX}" || "${CURRENT_PREFIX}" != "${DEFAULT_ENV_PATH}" ]]; then
    conda activate "${DEFAULT_ENV_PATH}" || true
  fi
fi

cd "${PROJECT_ROOT}"

${PYTHON_BIN} -m pip install --upgrade pip setuptools wheel
${PYTHON_BIN} -m pip install -r requirements.txt

if [[ -d "${PROJECT_ROOT}/thirdparty/diffusers" ]]; then
  ${PYTHON_BIN} -m pip install -e ./thirdparty/diffusers
fi

if [[ -d "${PROJECT_ROOT}/thirdparty/flash-attention-3" ]]; then
  echo "[INFO] flash-attention-3 source found at thirdparty/flash-attention-3"
  echo "[INFO] Build it manually only if your CUDA/PyTorch environment supports it."
fi

echo "[INFO] Environment ready."
