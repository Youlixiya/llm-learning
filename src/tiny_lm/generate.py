import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from .model import build_tiny_model


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def load_vocab(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    # JSON 中的键是字符串，需要转换回 int
    vocab["itos"] = {int(k): v for k, v in vocab["itos"].items()}
    return vocab


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos[i] for i in ids)


@torch.no_grad()
def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 20,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = PROCESSED_DIR / "tiny_lm_vocab.json"
    model_path = PROCESSED_DIR / "tiny_lm.pt"
    if not vocab_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型或词表，请先运行训练脚本：python src/tiny_lm/train.py\n"
            f"预计路径：{model_path}, {vocab_path}"
        )

    vocab = load_vocab(vocab_path)
    stoi = vocab["stoi"]
    itos = vocab["itos"]

    model = build_tiny_model(vocab_size=len(vocab["chars"]))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    input_ids = encode(prompt, stoi)
    if not input_ids:
        # 回退到一个默认字符
        first_char = next(iter(stoi.keys()))
        input_ids = [stoi[first_char]]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        if x.size(1) > model.max_seq_len:
            x = x[:, -model.max_seq_len :]

        _, last_logits = model(x, return_last_logits=True)
        logits = last_logits[0] / max(temperature, 1e-6)

        # top-k 过滤
        if top_k is not None and top_k > 0:
            values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
            mask = torch.full_like(logits, float("-inf"))
            mask[indices] = values
            logits = mask

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    generated_ids = x[0].tolist()
    return decode(generated_ids, itos)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Tiny Transformer 生成文本")
    parser.add_argument("--prompt", type=str, default="今天天气", help="起始提示语")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="生成的最大新 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_k", type=int, default=20, help="top-k 采样的 k 值")

    args = parser.parse_args()

    text = generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()

