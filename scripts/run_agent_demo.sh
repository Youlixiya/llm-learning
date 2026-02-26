#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "[Agent] 启动简单的命令行 Agent（支持 search_docs / calculator / get_time 工具）..."
python src/agents/simple_agent.py \
  --model_name_or_path "${MODEL_NAME}"

echo "[Agent] 结束。"

