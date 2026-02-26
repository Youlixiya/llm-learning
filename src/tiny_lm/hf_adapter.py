from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from .model import TinyTransformerLM


class TinyLMHFConfig(PretrainedConfig):
    model_type = "tiny_lm_rope"

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class TinyLMHFModel(PreTrainedModel):
    config_class = TinyLMHFConfig

    def __init__(self, config: TinyLMHFConfig) -> None:
        super().__init__(config)
        self.model = TinyTransformerLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        self.lm_head = nn.Identity()  # 与内部 head 共享，由 TinyTransformerLM 直接输出 logits

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.token_emb

    def set_input_embeddings(self, new_embeddings: nn.Module) -> None:
        self.model.token_emb = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits, _ = self.model(input_ids)

        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            vocab_size = logits.size(-1)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, vocab_size),
                labels.view(-1),
            )

        return {"loss": loss, "logits": logits}

    # ==== Generation API（兼容 HF CausalLM）====
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        # 目前没有做 KV cache，直接把全部 input_ids 喂给模型即可
        return {"input_ids": input_ids}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> torch.LongTensor:
        """简单封装一下 HF 的通用 generate，方便像 CausalLM 一样直接调用。"""
        return super().generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


def from_tiny_checkpoint(
    ckpt_path: str,
    vocab_size: int,
    config_kwargs: Optional[dict] = None,
) -> TinyLMHFModel:
    """从当前 tiny_lm 的 state_dict 加载，并返回 HuggingFace 风格模型。"""
    cfg = TinyLMHFConfig(vocab_size=vocab_size, **(config_kwargs or {}))
    model = TinyLMHFModel(cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.model.load_state_dict(state_dict)
    return model

