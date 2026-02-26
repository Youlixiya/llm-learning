import math
from typing import Optional, Tuple

import torch
from torch import nn


def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE 旋转位置编码（简化版，参考 LLaMA 等实现）。

    Args:
        q, k: (batch, heads, seq_len, head_dim)
        seq_len: 实际序列长度
    """
    bsz, n_heads, _, head_dim = q.size()
    device = q.device

    # 只在前半维度上做旋转，典型实现
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    # 这里直接采用较大的 base，兼容长序列扩展玩法
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, half_dim)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)

    cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, head_dim)
    sin = emb.sin()[None, None, :, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = q * cos + rotate(q) * sin
    k_rot = k * cos + rotate(k) * sin
    return q_rot, k_rot


class RoPEMultiHeadSelfAttention(nn.Module):
    """带 RoPE 的多头自注意力，用作“阿里 RoPE 注意力”教学版实现。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # 阿里 NeurIPS 最佳论文风格的门控：对注意力输出再做一层线性 + Sigmoid 作为 gate
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (batch, heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        q, k = _apply_rotary_pos_emb(q, k, seq_len=seq_len)

        # 缩放点积注意力 + 因果 mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=x.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_scores = attn_scores + causal_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        # 先做标准输出投影
        out = self.out_proj(attn_output)
        # 再做门控：门控分数依赖于当前 token 的注意力输出
        gate = torch.sigmoid(self.gate_proj(out))
        out = out * gate
        out = self.resid_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力残差
        x = x + self.attn(self.ln_1(x))
        # FFN 残差
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyTransformerLM(nn.Module):
    """带 RoPE 自注意力的 Tiny Transformer 语言模型（适配 HuggingFace / Ollama 的核心结构）。

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
        # 位置编码由 RoPE 提供，不再需要显式 pos embedding

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

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

        x = self.token_emb(input_ids)  # (batch, seq_len, d_model)

        for block in self.blocks:
            x = block(x)

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

