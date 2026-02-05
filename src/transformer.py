"""
Autoregressive Transformer for generating VQ-VAE token sequences,
conditioned on text input.

Generates a 16x16 = 256 token sequence representing codebook indices.

Optimized with:
- Flash Attention via F.scaled_dot_product_attention
- KV-cache for fast autoregressive generation
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Simple word-embedding based text encoder."""

    def __init__(self, vocab_size: int, embed_dim: int = 256, max_len: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.pos_embedding(pos)
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.dropout = dropout

        # Self-attention projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        # Cross-attention projections
        self.cross_q_proj = nn.Linear(dim, dim)
        self.cross_k_proj = nn.Linear(dim, dim)
        self.cross_v_proj = nn.Linear(dim, dim)
        self.cross_out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, text_ctx, causal_mask=None, kv_cache=None, use_cache=False):
        B, L, _ = x.shape

        # Self-attention with Flash Attention
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # KV-cache for generation
        if use_cache:
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        # Flash Attention (automatically used when available)
        h = F.scaled_dot_product_attention(q, k, v, is_causal=(causal_mask is None and not use_cache))
        h = h.transpose(1, 2).contiguous().view(B, L, self.dim)
        h = self.out_proj(h)
        x = x + h

        # Cross-attention to text (no causal mask needed)
        h = self.norm2(x)
        q = self.cross_q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k_proj(text_ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v_proj(text_ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).contiguous().view(B, L, self.dim)
        h = self.cross_out_proj(h)
        x = x + h

        # Feed-forward
        x = x + self.ff(self.norm3(x))
        return x, new_kv_cache


class TokenTransformer(nn.Module):
    """Autoregressive transformer that generates VQ token sequences conditioned on text."""

    def __init__(
        self,
        codebook_size: int = 512,
        seq_len: int = 256,
        dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        vocab_size: int = 1000,
        text_max_len: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.codebook_size = codebook_size
        self.dim = dim
        self.num_layers = num_layers

        # Text encoder
        self.text_encoder = TextEncoder(vocab_size, dim, text_max_len)

        # Token embeddings (codebook_size + 1 for BOS token)
        self.bos_token = codebook_size
        self.token_embedding = nn.Embedding(codebook_size + 1, dim)
        self.pos_embedding = nn.Embedding(seq_len + 1, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, codebook_size)

    def forward(self, indices, text_tokens):
        """
        Training forward pass (teacher forcing).

        Args:
            indices: (B, 256) ground-truth codebook indices
            text_tokens: (B, L) text token indices

        Returns:
            logits: (B, 256, codebook_size)
        """
        B = indices.shape[0]
        device = indices.device

        # Encode text
        text_ctx = self.text_encoder(text_tokens)

        # Prepend BOS
        bos = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)
        input_seq = torch.cat([bos, indices], dim=1)[:, :-1]

        # Embed tokens + positions
        pos = torch.arange(self.seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_seq) + self.pos_embedding(pos)

        # Transformer layers (causal via is_causal=True in scaled_dot_product_attention)
        for layer in self.layers:
            x, _ = layer(x, text_ctx, use_cache=False)

        x = self.norm(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, text_tokens, temperature: float = 1.0, top_k: int = 0):
        """
        Autoregressively generate a token sequence with KV-cache.

        Args:
            text_tokens: (B, L) text token indices
            temperature: sampling temperature
            top_k: if > 0, only sample from top-k logits

        Returns:
            (B, 256) generated codebook indices
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        text_ctx = self.text_encoder(text_tokens)

        # Initialize KV-caches for each layer
        kv_caches = [None] * self.num_layers

        # Start with BOS
        generated = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)

        for step in range(self.seq_len):
            # Only process the last token (use KV-cache for previous)
            if step == 0:
                # First step: process BOS
                pos = torch.zeros(B, 1, dtype=torch.long, device=device)
                x = self.token_embedding(generated) + self.pos_embedding(pos)
            else:
                # Subsequent steps: only process new token
                pos = torch.full((B, 1), step, dtype=torch.long, device=device)
                x = self.token_embedding(generated[:, -1:]) + self.pos_embedding(pos)

            # Transformer layers with KV-cache
            new_kv_caches = []
            for i, layer in enumerate(self.layers):
                x, new_kv = layer(x, text_ctx, kv_cache=kv_caches[i], use_cache=True)
                new_kv_caches.append(new_kv)
            kv_caches = new_kv_caches

            x = self.norm(x)
            logits = self.head(x[:, -1, :])

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]  # Remove BOS
