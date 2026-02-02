"""
Autoregressive Transformer for generating VQ-VAE token sequences,
conditioned on text input.

Generates a 4x4 = 16 token sequence representing codebook indices.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Simple word-embedding based text encoder."""

    def __init__(self, vocab_size: int, embed_dim: int = 512, max_len: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, L) token indices
        Returns:
            (B, L, D) text embeddings
        """
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.pos_embedding(pos)
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        # Causal self-attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        # Cross-attention to text
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
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

    def forward(self, x, text_ctx, causal_mask=None):
        # Self-attention (causal)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask)
        x = x + h

        # Cross-attention to text
        h = self.norm2(x)
        h, _ = self.cross_attn(h, text_ctx, text_ctx)
        x = x + h

        # Feed-forward
        x = x + self.ff(self.norm3(x))
        return x


class TokenTransformer(nn.Module):
    """Autoregressive transformer that generates VQ token sequences conditioned on text."""

    def __init__(
        self,
        codebook_size: int = 512,
        seq_len: int = 16,  # 4x4
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 1000,
        text_max_len: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.codebook_size = codebook_size
        self.dim = dim

        # Text encoder
        self.text_encoder = TextEncoder(vocab_size, dim, text_max_len)

        # Token embeddings (codebook_size + 1 for BOS token)
        self.bos_token = codebook_size  # BOS = codebook_size
        self.token_embedding = nn.Embedding(codebook_size + 1, dim)
        self.pos_embedding = nn.Embedding(seq_len + 1, dim)  # +1 for BOS

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, codebook_size)

    def _causal_mask(self, length, device):
        mask = torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()
        return mask

    def forward(self, indices, text_tokens):
        """
        Training forward pass.

        Args:
            indices: (B, 16) ground-truth codebook indices
            text_tokens: (B, L) text token indices

        Returns:
            logits: (B, 16, codebook_size)
        """
        B = indices.shape[0]
        device = indices.device

        # Encode text
        text_ctx = self.text_encoder(text_tokens)

        # Prepend BOS
        bos = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)
        input_seq = torch.cat([bos, indices], dim=1)[:, :-1]  # (B, 16)

        # Embed tokens + positions
        pos = torch.arange(self.seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_seq) + self.pos_embedding(pos)

        # Causal mask
        mask = self._causal_mask(self.seq_len, device)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, text_ctx, causal_mask=mask)

        x = self.norm(x)
        logits = self.head(x)  # (B, 16, codebook_size)
        return logits

    @torch.no_grad()
    def generate(self, text_tokens, temperature: float = 1.0, top_k: int = 0):
        """
        Autoregressively generate a token sequence.

        Args:
            text_tokens: (B, L) text token indices
            temperature: sampling temperature
            top_k: if > 0, only sample from top-k logits

        Returns:
            (B, 16) generated codebook indices
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        text_ctx = self.text_encoder(text_tokens)

        # Start with BOS
        generated = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)

        for step in range(self.seq_len):
            seq_len_cur = generated.shape[1]
            pos = torch.arange(seq_len_cur, device=device).unsqueeze(0)
            x = self.token_embedding(generated) + self.pos_embedding(pos)

            mask = self._causal_mask(seq_len_cur, device)
            for layer in self.layers:
                x = layer(x, text_ctx, causal_mask=mask)

            x = self.norm(x)
            logits = self.head(x[:, -1, :])  # (B, codebook_size)

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]  # Remove BOS, (B, 16)
