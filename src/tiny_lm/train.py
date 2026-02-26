from dataclasses import dataclass
from pathlib import Path

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
    model_max_length: int = 128
    batch_size: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 3e-4
    logging_steps: int = 10
    save_steps: int = 100
    output_dir: str = "tiny_lm_hf"
    base_tokenizer: str = "Qwen/Qwen2.5-0.5B"


def build_corpus() -> str:
    # 一个简单的中英混合玩具语料，用于演示训练流程
    corpus = """
今天天气很好，适合学习大语言模型。
大语言模型可以用于对话、问答、总结和代码生成。
只要掌握了基本原理和工程套路，你就能把 LLM 用在自己的项目里。

Today is a good day to learn about large language models.
With a tiny Transformer, we can train a toy language model from scratch.
This model will not be very strong, but it is perfect for education purposes.
"""
    # 去掉两端空白
    return corpus.strip()


def train() -> None:
    cfg = TinyLMTrainerConfig()

    # 1) 使用 Qwen tokenizer（或其它 HF tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_tokenizer,
        trust_remote_code=True,
    )
    # 为了兼容 tiny 模型的 max length
    tokenizer.model_max_length = cfg.model_max_length

    # 2) TinyLM HF 模型，vocab_size 与 tokenizer 对齐
    model_config = TinyLMHFConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cfg.model_max_length,
    )
    model = TinyLMHFModel(model_config)

    # 3) 构造一个最小示例语料，并转成 HF Dataset
    corpus = build_corpus()
    dataset = Dataset.from_dict({"text": [corpus]})

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


if __name__ == "__main__":
    train()

