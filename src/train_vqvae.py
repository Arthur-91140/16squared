"""
Stage 1: Train VQ-VAE on Minecraft item textures.

The VQ-VAE learns to encode/decode 16x16 RGBA textures through a discrete codebook.
This must be trained before the Transformer (Stage 2).

Usage:
    python -m src.train_vqvae --data_dir data/raw --epochs 100
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import create_dataloader
from .vqvae import VQVAE


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def save_samples(model, dataloader, device, epoch, output_dir):
    """Save a grid of original vs reconstructed textures."""
    from torchvision.utils import save_image

    model.eval()
    batch = next(iter(dataloader))
    images = batch["image"][:8].to(device)

    with torch.no_grad():
        recon, _, _, _ = model(images)

    # Denormalize from [-1,1] to [0,1]
    images = (images + 1) / 2
    recon = (recon + 1) / 2

    comparison = torch.cat([images, recon], dim=0)
    save_image(comparison, os.path.join(output_dir, f"vqvae_epoch_{epoch:04d}.png"), nrow=8)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE (Stage 1)")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataloader, dataset = create_dataloader(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Dataset: {len(dataset)} textures")

    # Model
    model = VQVAE(
        in_channels=4,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_embeddings=args.codebook_size,
        commitment_cost=args.commitment_cost,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Logging
    writer = SummaryWriter(os.path.join(args.output_dir, "logs_vqvae"))

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_recon = 0
        total_commit = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)

            recon, recon_loss, commit_loss, _ = model(images)
            loss = recon_loss + commit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_commit += commit_loss.item()
            n_batches += 1

            pbar.set_postfix(recon=f"{recon_loss.item():.4f}", commit=f"{commit_loss.item():.4f}")

        avg_recon = total_recon / n_batches
        avg_commit = total_commit / n_batches
        writer.add_scalar("loss/recon", avg_recon, epoch)
        writer.add_scalar("loss/commit", avg_commit, epoch)
        print(f"Epoch {epoch+1}: recon={avg_recon:.4f}, commit={avg_commit:.4f}")

        # Save samples
        save_samples(model, dataloader, device, epoch, args.output_dir)

        # Checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model, optimizer, epoch,
                avg_recon + avg_commit,
                os.path.join(args.checkpoint_dir, f"vqvae_epoch_{epoch+1:04d}.pt"),
            )

    writer.close()
    print("VQ-VAE training complete.")


if __name__ == "__main__":
    main()
