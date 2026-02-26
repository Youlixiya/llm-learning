import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import build_tiny_model


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TinyLMConfig:
    block_size: int = 64
    batch_size: int = 32
    max_iters: int = 1000
    eval_interval: int = 200
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CharDataset(Dataset):
    def __init__(self, data_ids: List[int], block_size: int) -> None:
        super().__init__()
        self.data = torch.tensor(data_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


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


def build_vocab(text: str) -> Dict[str, Dict]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab = {
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
    }
    return vocab


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text]


def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos[i] for i in ids)


def save_vocab(vocab: Dict, path: Path) -> None:
    # 将键转换为字符串，便于 JSON 序列化
    stoi = vocab["stoi"]
    itos = vocab["itos"]
    vocab_to_save = {
        "chars": vocab["chars"],
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)


def train() -> None:
    cfg = TinyLMConfig()
    device = torch.device(cfg.device)

    corpus = build_corpus()
    vocab = build_vocab(corpus)
    stoi = vocab["stoi"]
    itos = vocab["itos"]
    data_ids = encode(corpus, stoi)

    dataset = CharDataset(data_ids, block_size=cfg.block_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = build_tiny_model(vocab_size=len(vocab["chars"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    step = 0
    for epoch in range(10):  # 简单跑若干轮，配合 max_iters 控制总步数
        for x, y in dataloader:
            step += 1
            if step > cfg.max_iters:
                break

            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            # (batch, seq, vocab) -> (batch * seq, vocab)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg.eval_interval == 0:
                print(f"[step {step}] train loss = {loss.item():.4f}")
                # 简单生成一小段文本观察效果
                print(sample_text(model, stoi, itos, device=device))
                print("-" * 40)

        if step > cfg.max_iters:
            break

    # 保存模型与词表
    model_path = PROCESSED_DIR / "tiny_lm.pt"
    vocab_path = PROCESSED_DIR / "tiny_lm_vocab.json"
    torch.save(model.state_dict(), model_path)
    save_vocab(vocab, vocab_path)

    print(f"模型已保存到: {model_path}")
    print(f"词表已保存到: {vocab_path}")


@torch.no_grad()
def sample_text(
    model: nn.Module,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    prompt: str = "今天天气",
    max_new_tokens: int = 100,
) -> str:
    model.eval()
    # 如果 prompt 中包含未见过的字符，简单跳过
    input_ids = [stoi[ch] for ch in prompt if ch in stoi]
    if not input_ids:
        # 回退到一个默认字符
        first_char = next(iter(stoi.keys()))
        input_ids = [stoi[first_char]]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        if x.size(1) > model.max_seq_len:
            x = x[:, -model.max_seq_len :]

        _, last_logits = model(x, return_last_logits=True)
        probs = torch.softmax(last_logits[0], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    generated_ids = x[0].tolist()
    return decode(generated_ids, itos)


if __name__ == "__main__":
    train()

