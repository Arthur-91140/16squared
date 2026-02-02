"""
VQ-VAE for 16x16 RGBA Minecraft textures.

Architecture:
    Encoder: 16x16x4 -> 4x4x256 (via 2 downsampling blocks)
    Codebook: 512 entries, dimension 256
    Decoder: 4x4x256 -> 16x16x4 (via 2 upsampling blocks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 16x16 -> 8x8
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            ResBlock(hidden_dim),
            # 8x8 -> 4x4
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.GELU(),
            ResBlock(hidden_dim * 2),
            # Project to embedding dim
            nn.Conv2d(hidden_dim * 2, embed_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 4, hidden_dim: int = 128, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # Project from embedding dim
            nn.Conv2d(embed_dim, hidden_dim * 2, 1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.GELU(),
            ResBlock(hidden_dim * 2),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            ResBlock(hidden_dim),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    """Vector Quantization with EMA updates."""

    def __init__(self, num_embeddings: int = 512, embed_dim: int = 256, commitment_cost: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())

    def forward(self, z):
        # z: (B, D, H, W)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)

        # Distances to codebook
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )

        indices = distances.argmin(dim=1)  # (B*H*W,)
        z_q = self.embedding(indices).view(B, H, W, D).permute(0, 3, 1, 2)

        # EMA update
        if self.training:
            encodings = F.one_hot(indices, self.num_embeddings).float()
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                encodings.sum(0), alpha=1 - self.ema_decay
            )
            embed_sum = z_flat.t() @ encodings  # (D, K)
            self.ema_embed_sum.mul_(self.ema_decay).add_(
                embed_sum.t(), alpha=1 - self.ema_decay
            )
            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.num_embeddings * 1e-5)
                * n
            )
            self.embedding.weight.data.copy_(
                self.ema_embed_sum / cluster_size.unsqueeze(1)
            )

        # Loss
        commitment_loss = F.mse_loss(z, z_q.detach())
        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        indices_grid = indices.view(B, H, W)
        return z_q_st, commitment_loss, indices_grid

    def decode_indices(self, indices):
        """Convert codebook indices back to embeddings.

        Args:
            indices: (B, H, W) tensor of codebook indices
        Returns:
            (B, D, H, W) tensor of quantized embeddings
        """
        B, H, W = indices.shape
        flat = indices.reshape(-1)
        z_q = self.embedding(flat)
        return z_q.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, embed_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embed_dim, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_dim, embed_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, commit_loss, indices = self.quantizer(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, recon_loss, commit_loss, indices

    def encode(self, x):
        """Encode images to codebook indices."""
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices

    def decode(self, indices):
        """Decode codebook indices to images."""
        z_q = self.quantizer.decode_indices(indices)
        return self.decoder(z_q)
