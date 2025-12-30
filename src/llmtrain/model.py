from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyGPTConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 128
    activation_checkpointing: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ToyGPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask buffer
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config: ToyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.activation_checkpointing = config.activation_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _forward(x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

        if self.activation_checkpointing and x.requires_grad:
            # reduce memory usage for deeper models
            return torch.utils.checkpoint.checkpoint(_forward, x)
        return _forward(x)


class ToyGPTModel(nn.Module):
    def __init__(self, config: Optional[ToyGPTConfig] = None):
        super().__init__()
        self.config = config or ToyGPTConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.pos_embedding = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layers)])
        self.ln_f = nn.LayerNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.size()
        if T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss
