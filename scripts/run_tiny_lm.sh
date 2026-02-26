#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "[Tiny LM] 开始训练 Tiny Transformer 语言模型..."
python src/tiny_lm/train.py

echo "[Tiny LM] 使用训练好的模型生成示例文本..."
python src/tiny_lm/generate.py --prompt "今天天气"

echo "[Tiny LM] 完成。"

