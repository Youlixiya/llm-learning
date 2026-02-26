#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "[RAG] 构建示例知识库索引..."
python src/rag/build_index.py

echo "[RAG] 在知识库上进行一次示例问答..."
python src/rag/query_rag.py \
  --model_name_or_path "${MODEL_NAME}" \
  --question "这套 LLM 教程推荐的整体学习路线是什么？" \
  --top_k 2

echo "[RAG] 完成。"

