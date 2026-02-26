from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
from typing import Any, Dict

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .hf_adapter import TinyLMHFConfig, TinyLMHFModel


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TinyLMTrainerConfig:
    # ---- 数据 / 训练相关 ----
    model_max_length: int = 128
    batch_size: int = 8
    num_train_epochs: int = 1  # 只训练 1 个 epoch
    learning_rate: float = 3e-4
    logging_steps: int = 10
    save_steps: int = 100
    output_dir: str = "tiny_lm_hf"

    # ---- tokenizer 相关 ----
    base_tokenizer: str = "Qwen/Qwen2.5-0.5B"

    # ---- 模型规模相关（更大一些，适配 5090 显卡）----
    d_model: int = 512
    n_heads: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    @classmethod
    def from_json(cls, path: str) -> "TinyLMTrainerConfig":
        """从 JSON 配置文件加载，并在 dataclass 默认值基础上覆盖。"""
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        default_dict = asdict(cls())
        default_dict.update(data)
        return cls(**default_dict)


def train(cfg: TinyLMTrainerConfig) -> None:

    # 1) 使用 Qwen tokenizer（或其它 HF tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_tokenizer,
        trust_remote_code=True,
    )
    # 为了兼容 tiny 模型的 max length
    tokenizer.model_max_length = cfg.model_max_length

    # 2) TinyLM HF 模型，vocab_size 与 tokenizer 对齐
    # 注意：很多 HF tokenizer（包括 Qwen）会有额外的 added_tokens，
    # tokenizer.vocab_size 不包含这些，而编码后的 token id 可能会用到它们，
    # 所以这里必须用 len(tokenizer) 来作为真正的 vocab_size，避免 embedding / CE 越界。
    model_config = TinyLMHFConfig(
        vocab_size=len(tokenizer),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        max_seq_len=cfg.model_max_length,
        dropout=cfg.dropout,
    )
    model = TinyLMHFModel(model_config)

    # 3) 从本地 jsonl 预训练数据构造 HF Dataset
    pretrain_path = DATA_DIR / "pretrain_hq.jsonl"
    if not pretrain_path.exists():
        raise FileNotFoundError(f"预训练数据文件不存在: {pretrain_path}")

    # jsonl 中每一行是 {"text": "..."}，直接用 datasets 加载
    dataset = Dataset.from_json(str(pretrain_path))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.model_max_length,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    # 4) 自回归语言模型的数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) HF Trainer
    output_dir = PROCESSED_DIR / cfg.output_dir
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_strategy="epoch",  # 每个 epoch 结束时保存
        save_total_limit=1,  # 只保留最后一次 checkpoint
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"HF Trainer 训练完成，模型与 tokenizer 已保存到: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyLM with HF Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件路径（可选），用于覆盖默认 TinyLMTrainerConfig。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        cfg = TinyLMTrainerConfig.from_json(args.config)
        print(f"使用外部配置文件: {args.config}")
    else:
        cfg = TinyLMTrainerConfig()
        print("使用 TinyLMTrainerConfig 默认配置。")

    train(cfg)

