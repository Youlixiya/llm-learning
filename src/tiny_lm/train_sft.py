import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .hf_adapter import TinyLMHFConfig, TinyLMHFModel


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TinyLMSFTConfig:
    # ---- 数据 / 训练相关 ----
    model_max_length: int = 512
    batch_size: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 5e-4
    logging_steps: int = 10
    save_steps: int = 100
    output_dir: str = "tiny_lm_sft"

    # SFT 数据（默认使用仓库自带的 mini 数据集）
    dataset_path: str = str(DATA_DIR / "sft_mini_512.jsonl")

    # ---- tokenizer / 模型相关 ----
    # 用 Qwen Instruct 版本的 tokenizer，方便直接使用其 chat_template
    base_tokenizer: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # 预训练好的 TinyLM HF checkpoint（如果存在则从这里继续训练）
    base_model_dir: str = str(PROCESSED_DIR / "tiny_lm_hf")

    # Tiny 模型规模（如果找不到预训练权重，则用这些超参随机初始化）
    d_model: int = 512
    n_heads: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    @classmethod
    def from_json(cls, path: str) -> "TinyLMSFTConfig":
        """从 JSON 配置文件加载，并在 dataclass 默认值基础上覆盖。"""
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        default_dict = asdict(cls())
        default_dict.update(data)
        return cls(**default_dict)


def load_sft_dataset(path: Path) -> Dataset:
    """从 JSONL 文件加载指令 / SFT 数据。

    约定每行 JSON 至少包含：
    - instruction: 用户指令
    - input: 可选补充输入
    - output / response: 模型期望回答
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到 SFT 数据文件：{path}")

    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {
                    "instruction": obj.get("instruction", ""),
                    "input": obj.get("input", "") or "",
                    "output": obj.get("output") or obj.get("response") or "",
                }
            )

    if not records:
        raise ValueError(f"SFT 数据文件为空或格式不正确：{path}")

    return Dataset.from_list(records)


def format_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
) -> Dict[str, Any]:
    """将单条指令样本转换为适配 Qwen Chat 模板的 token 序列。

    这里使用 Qwen 的 chat_template 来构造对话：
    - user: instruction (+ 可选 input)
    - assistant: output

    这样生成的文本中会自动包含 Qwen 的特殊标记（例如 <|im_start|> / <|im_end|>），
    模型在 SFT 过程中会学习到“对话轮次的开始和结束”，
    后续推理时就可以用多轮对话的方式与模型交互。
    """
    user_content = example["instruction"].strip()
    if example.get("input"):
        user_content += "\n\n输入：\n" + str(example["input"]).strip()

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": str(example["output"]).strip()},
    ]

    # 使用 Qwen 的 chat_template 生成带特殊标记的完整对话文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    # 监督微调：输入和标签完全对齐
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def build_model_and_tokenizer(cfg: TinyLMSFTConfig) -> tuple[TinyLMHFModel, Any]:
    """加载 tokenizer，并根据情况构造 / 加载 TinyLMHF 模型。"""
    # 使用 Qwen Instruct tokenizer，带有 chat_template 和特殊 token
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_tokenizer,
        trust_remote_code=True,
    )
    tokenizer.model_max_length = cfg.model_max_length

    base_model_path = Path(cfg.base_model_dir)
    if (base_model_path / "config.json").exists():
        print(f"从预训练 TinyLM HF checkpoint 加载模型：{base_model_path}")
        model = TinyLMHFModel.from_pretrained(str(base_model_path))
        # 安全起见，检查 vocab_size 是否与当前 tokenizer 一致
        if model.config.vocab_size != len(tokenizer):
            raise ValueError(
                f"预训练 TinyLM vocab_size={model.config.vocab_size} "
                f"与当前 tokenizer 大小 {len(tokenizer)} 不一致，请确认使用了同一套 tokenizer。"
            )
    else:
        print(
            "未找到预训练 TinyLM HF checkpoint，将使用随机初始化权重进行 SFT。\n"
            f"预期 checkpoint 位置：{base_model_path}"
        )
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

    return model, tokenizer


def train_sft(cfg: TinyLMSFTConfig) -> None:
    # 1) 模型与 tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg)

    # 2) 加载 SFT 指令数据
    dataset_path = Path(cfg.dataset_path)
    raw_dataset = load_sft_dataset(dataset_path)
    print(f"加载 SFT 指令样本 {len(raw_dataset)} 条，用于 TinyLM 对话微调。")

    def _format_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return format_example(example, tokenizer, cfg.model_max_length)

    tokenized_dataset = raw_dataset.map(
        _format_fn,
        remove_columns=raw_dataset.column_names,
    )

    # 3) HF Trainer 配置
    output_dir = PROCESSED_DIR / cfg.output_dir
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"TinyLM SFT 训练完成，模型与 tokenizer 已保存到: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 SFT 指令数据，对 TinyLM 进行多轮对话微调"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件路径（可选），用于覆盖默认 TinyLMSFTConfig。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        cfg = TinyLMSFTConfig.from_json(args.config)
        print(f"使用外部配置文件: {args.config}")
    else:
        cfg = TinyLMSFTConfig()
        print("使用 TinyLMSFTConfig 默认配置。")

    train_sft(cfg)

