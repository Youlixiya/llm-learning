#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

OUT_DIR="${1:-tiny_lm_hf}"
export OUT_DIR

python - << 'PYCODE'
import json
from pathlib import Path

import torch

from src.tiny_lm.hf_adapter import TinyLMHFConfig, TinyLMHFModel, from_tiny_checkpoint
from src.tiny_lm.train import PROCESSED_DIR
from src.tiny_lm.train import build_vocab, build_corpus  # 复用现有词表逻辑

import os

root = Path(__file__).resolve().parents[1]
out_dir = root / os.environ["OUT_DIR"]
out_dir.mkdir(parents=True, exist_ok=True)

corpus = build_corpus()
vocab = build_vocab(corpus)
vocab_size = len(vocab["chars"])

ckpt_path = PROCESSED_DIR / "tiny_lm.pt"
model = from_tiny_checkpoint(str(ckpt_path), vocab_size=vocab_size)

config = model.config
config.save_pretrained(out_dir)
model.save_pretrained(out_dir)

# 同时保存 tokenizer-like 映射，Ollama / 自定义加载时会用到
with (out_dir / "tiny_vocab.json").open("w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"Saved HuggingFace style model to: {out_dir}")
PYCODE

