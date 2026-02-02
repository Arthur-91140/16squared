"""
Centralized configuration for all hyperparameters.
"""

from dataclasses import dataclass, field


@dataclass
class VQVAEConfig:
    in_channels: int = 4
    hidden_dim: int = 128
    embed_dim: int = 256
    codebook_size: int = 512
    commitment_cost: float = 0.25
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 100
    save_every: int = 10


@dataclass
class TransformerConfig:
    codebook_size: int = 512
    seq_len: int = 16
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    text_max_len: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 200
    save_every: int = 20


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 50
    alpha_threshold: int = 128
    palette_levels: int = 16
