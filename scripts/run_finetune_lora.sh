#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/finetune/qwen25-0_5b-lora}"

echo "[Finetune] 使用 LoRA 对 ${MODEL_NAME} 进行小规模指令微调..."
python src/finetune/train_lora.py \
  --model_name_or_path "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS:-50}"

echo "[Finetune] 使用微调后的模型进行一次示例对话..."
python src/finetune/infer_lora.py \
  --model_name_or_path "${MODEL_NAME}" \
  --adapter_dir "${OUTPUT_DIR}" \
  --prompt "请用 3 条要点总结：为什么要系统学习大语言模型工程？"

echo "[Finetune] 完成。"

