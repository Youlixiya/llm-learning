import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LoraTrainConfig:
    model_name_or_path: str
    output_dir: Path
    dataset_path: Optional[Path]
    max_steps: int = 50
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    max_seq_length: int = 512


def build_tiny_instruct_dataset() -> Dataset:
    """构造一小批内置的中文指令数据，方便开箱即用。"""
    examples: List[Dict[str, str]] = [
        {
            "instruction": "用三句话总结下面的文本。",
            "input": (
                "大语言模型（LLM）在近年来迅速发展，已经成为代码生成、问答和写作的重要工具。"
                "工程师需要理解其基本原理和工程实践，才能真正把模型用到自己的业务中去。"
            ),
            "output": (
                "1. 大语言模型已经成为通用智能的重要基础设施。\n"
                "2. 仅仅会“调接口”远远不够，还需要理解其原理和工程约束。\n"
                "3. 掌握 LLM 工程能力，可以极大提升个人和团队的生产力。"
            ),
        },
        {
            "instruction": "你现在是一个中文技术写作助手，请帮我写一段学习 LLM 的动机说明。",
            "input": "",
            "output": (
                "学习大语言模型并不是追逐热点，而是掌握一项未来多年都会持续演进的基础能力。"
                "无论是自动化日常工作、构建智能应用，还是升级自己的思考方式，LLM 都可以发挥巨大价值。"
            ),
        },
        {
            "instruction": "请把下面的待办事项整理成 3 条有优先级的任务列表。",
            "input": "学习 Transformer 原理；搭建一个 RAG 知识库；在自己项目中集成一个对话助手。",
            "output": (
                "1. 高优先级：在现有项目中集成一个对话助手，直接产生业务价值。\n"
                "2. 中优先级：搭建一个简单的 RAG 知识库，让助手具备查询文档的能力。\n"
                "3. 低优先级：系统学习 Transformer 原理，为后续深入优化和定制打基础。"
            ),
        },
        {
            "instruction": "你是一个严谨的技术面试官，根据下面岗位说明，生成 3 个面试问题。",
            "input": "岗位：LLM 应用工程师，需要掌握 Python、PyTorch，以及基本的 RAG 和指令微调技能。",
            "output": (
                "1. 请你结合一个具体项目，说明如何从零搭建一个 RAG 系统，包括数据准备、索引和检索流程。\n"
                "2. 在资源有限的情况下，你会如何选择基座模型并设计 LoRA 微调方案？\n"
                "3. 你如何评估一个 LLM 应用的效果？在没有标准答案的数据集时，会采用什么样的评测策略？"
            ),
        },
    ]
    return Dataset.from_list(examples)


def load_instruct_dataset(path: Optional[Path]) -> Dataset:
    """从 JSONL 文件加载指令数据，或回退到内置示例数据。"""
    if path is None:
        print("未指定 dataset_path，使用内置的示例指令数据。")
        return build_tiny_instruct_dataset()

    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

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
        raise ValueError(f"数据文件为空或格式不正确：{path}")

    return Dataset.from_list(records)


def format_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
) -> Dict[str, Any]:
    """将单条指令样本转换为适配 Chat 模型的 token 序列。"""
    user_content = example["instruction"].strip()
    if example.get("input"):
        user_content += "\n\n输入：\n" + str(example["input"]).strip()

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": str(example["output"]).strip()},
    ]

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


def get_lora_model(
    model_name_or_path: str,
) -> Any:
    """加载基座模型并应用 LoRA 适配器配置。"""
    print(f"加载基座模型：{model_name_or_path}")
    device_map = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # 通用的 LoRA 配置，针对所有线性层启用适配器，便于兼容不同架构
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def parse_args() -> LoraTrainConfig:
    parser = argparse.ArgumentParser(description="使用 LoRA 对开源 Chat 模型进行指令微调")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face 上的基座模型名称或本地路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROCESSED_DIR / "finetune" / "qwen25-0_5b-lora"),
        help="LoRA 适配器的保存路径",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="指令数据集的 JSONL 路径（可选），不指定则使用内置示例数据",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="训练的最大 step 数（示例默认较小，适合快速试跑）",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="每个设备上的 batch size",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="单条样本的最大 token 长度",
    )

    args = parser.parse_args()
    return LoraTrainConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=Path(args.output_dir),
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_seq_length=args.max_seq_length,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_instruct_dataset(cfg.dataset_path)
    print(f"加载指令样本 {len(raw_dataset)} 条，用于 LoRA 微调。")

    def _format_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return format_example(example, tokenizer, cfg.max_seq_length)

    tokenized_dataset = raw_dataset.map(_format_fn, remove_columns=raw_dataset.column_names)

    model = get_lora_model(cfg.model_name_or_path)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        save_steps=cfg.max_steps,
        save_total_limit=1,
        optim="adamw_torch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("训练完成，保存 LoRA 适配器权重...")

    model.save_pretrained(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    print(f"LoRA 适配器与分词器已保存到：{cfg.output_dir}")


if __name__ == "__main__":
    main()

