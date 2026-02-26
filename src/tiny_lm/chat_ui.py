import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from .hf_adapter import TinyLMHFModel


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_model_and_tokenizer(
    model_dir: str | None = None,
) -> Tuple[TinyLMHFModel, Any, torch.device]:
    """从 HF checkpoint 加载 TinyLM 模型和 tokenizer，用于推理 / 对话。

    默认从 SFT 输出目录 `data/processed/tiny_lm_sft` 加载，
    你也可以通过命令行参数 --model_dir 指定其它 checkpoint。
    """
    if model_dir is None:
        model_dir = str(PROCESSED_DIR / "tiny_lm_sft")

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型目录 {model_path}，"
            "请先运行 `train_sft.py` 完成 SFT 训练，或通过 --model_dir 指定已有 checkpoint。"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = TinyLMHFModel.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # eod / 停止 token：优先使用 tokenizer.eos_token_id，其次尝试 Qwen 的 <|im_end|>
    if tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id
    else:
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_token_id is None:
            eos_token_id = None  # 回退到无显式 eos

    # 将停止 id 挂在 tokenizer 上，后面直接使用
    tokenizer._tiny_eos_token_id = eos_token_id  # type: ignore[attr-defined]

    return model, tokenizer, device


def build_messages_from_history(
    history: List[Tuple[str, str]],
    last_user_message: str | None = None,
) -> List[Dict[str, str]]:
    """将 (user, assistant) 历史转换为 Qwen 风格的 messages 列表。"""
    messages: List[Dict[str, str]] = []
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    if last_user_message is not None:
        messages.append({"role": "user", "content": last_user_message})

    return messages


def generate_response(
    model: TinyLMHFModel,
    tokenizer,
    device: torch.device,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """给定多轮 messages，使用 Qwen chat_template 生成模型回复。"""
    # 1) 使用 chat_template 构造带特殊标记的对话文本，并追加 generation prompt
    prompt_text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2) 编码并送入 TinyLM
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)

    eos_token_id = getattr(tokenizer, "_tiny_eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 3) 解码，并截掉 prompt 部分，只保留模型新生成的回复
    full_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=False,
    )
    response_text = full_text[len(prompt_text) :].strip()
    return response_text


# ==========================
#  命令行多轮对话 CLI
# ==========================


def run_cli(
    model_dir: str | None = None,
    max_new_tokens: int = 128,
) -> None:
    model, tokenizer, device = load_model_and_tokenizer(model_dir)

    print("TinyLM CLI 对话已启动。")
    print("输入内容并回车与模型对话。输入 `/reset` 重置对话，`/exit` 退出。")

    history: List[Tuple[str, str]] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出 CLI 对话。")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("再见！")
            break

        if user_input.lower() in {"/reset", "reset"}:
            history.clear()
            print("对话历史已重置。")
            continue

        messages = build_messages_from_history(history, user_input)

        try:
            reply = generate_response(
                model,
                tokenizer,
                device,
                messages,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[错误] 生成回复失败：{e}")
            continue

        history.append((user_input, reply))
        print(f"模型：{reply}")


# ==========================
#  Gradio WebUI 多轮对话
# ==========================


def launch_gradio(
    model_dir: str | None = None,
    host: str = "0.0.0.0",
    port: int = 7860,
    max_new_tokens: int = 128,
) -> None:
    import gradio as gr

    model, tokenizer, device = load_model_and_tokenizer(model_dir)

    def chat_fn(message: str, history: List[Tuple[str, str]]):
        # history: List of [user, assistant]
        messages = build_messages_from_history(history, message)
        reply = generate_response(
            model,
            tokenizer,
            device,
            messages,
            max_new_tokens=max_new_tokens,
        )
        history.append((message, reply))
        return "", history

    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        title="TinyLM 多轮对话 WebUI",
        description="基于自定义 TinyLM + Qwen tokenizer 的轻量级多轮对话界面。",
        retry_btn="重试本轮",
        undo_btn="撤销上轮",
        clear_btn="清空对话",
    )

    chat_interface.launch(
        server_name=host,
        server_port=port,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TinyLM 多轮对话（CLI / Gradio WebUI）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["cli", "gradio"],
        help="运行模式：cli（命令行对话）或 gradio（WebUI）。默认 cli。",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="TinyLM HF checkpoint 目录（默认为 data/processed/tiny_lm_sft）。",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Gradio WebUI 绑定地址，默认 0.0.0.0。",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio WebUI 端口，默认 7860。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="单轮生成的最大新 token 数。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "cli":
        run_cli(
            model_dir=args.model_dir,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        launch_gradio(
            model_dir=args.model_dir,
            host=args.host,
            port=args.port,
            max_new_tokens=args.max_new_tokens,
        )

