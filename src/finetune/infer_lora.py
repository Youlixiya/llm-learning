import argparse
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


def chat_with_lora(
    model_name_or_path: str,
    adapter_dir: Path,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载基座模型：{model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"加载 LoRA 适配器：{adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.eval()

    messages: List[dict] = [
        {
            "role": "system",
            "content": "你是一个中文技术助手，请结合自己的知识和用户的描述，给出清晰、结构化的回答。",
        },
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0, inputs.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 LoRA 微调后的模型进行对话")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face 上的基座模型名称或本地路径",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="LoRA 适配器的保存路径（train_lora.py 的 output_dir）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请用三句话说明：为什么工程师要学习大语言模型？",
        help="用户输入的指令/问题",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="生成的最大新 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="采样温度",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="nucleus sampling 的 top-p 值",
    )

    args = parser.parse_args()

    text = chat_with_lora(
        model_name_or_path=args.model_name_or_path,
        adapter_dir=Path(args.adapter_dir),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()

