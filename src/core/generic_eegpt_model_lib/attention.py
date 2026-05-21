import torch.nn as nn
from .modelMethods import apply_rotary_emb
import torch
import math

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False,
                 return_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.return_attention = return_attention

    def forward(self, x, freqs=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3,B,nh,t,d
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,nh,t,d

        if self.use_rope:  # RoPE
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
                attn_maak = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_maak.masked_fill(torch.logical_not(attn_mask), -float('inf'))
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            else:
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            return attn_weight
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal)
        x = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, hs*nh)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x