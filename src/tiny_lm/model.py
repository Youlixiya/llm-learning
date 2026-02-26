import math
from typing import Optional, Tuple

import torch
from torch import nn


class TinyTransformerLM(nn.Module):
    """一个极简的 Transformer 语言模型，用于教学演示。

    - 输入：token id 序列，形状 (batch_size, seq_len)
    - 输出：对每个位置下一个 token 的 logits，形状 (batch_size, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 上三角为 -inf，保证只能看到当前位置及之前的 token
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        return_last_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            return_last_logits: 是否额外返回最后一个位置的 logits（用于生成）

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            last_logits: (batch_size, vocab_size) or None
        """
        bsz, seq_len = input_ids.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} 超过了 max_seq_len {self.max_seq_len}")

        device = input_ids.device
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)  # (1, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = self._generate_causal_mask(seq_len, device=device)
        x = self.encoder(x, mask=causal_mask)
        x = self.ln_f(x)

        logits = self.head(x)  # (bsz, seq_len, vocab_size)

        last_logits = logits[:, -1, :] if return_last_logits else None
        return logits, last_logits


def build_tiny_model(vocab_size: int) -> TinyTransformerLM:
    """构建一个默认配置的 Tiny Transformer 模型，方便在训练脚本中复用。"""
    return TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=512,
        max_seq_len=128,
        dropout=0.1,
    )

