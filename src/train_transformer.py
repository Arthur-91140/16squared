"""
Stage 2: Train autoregressive Transformer conditioned on text.

Requires a trained VQ-VAE checkpoint from Stage 1.
The Transformer learns to generate VQ token sequences from text prompts.

Usage:
    python -m src.train_transformer --data_dir data/raw --vqvae_ckpt checkpoints/vqvae_epoch_0100.pt
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import create_dataloader
from .vqvae import VQVAE
from .transformer import TokenTransformer


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def generate_samples(transformer, vqvae, dataset, device, epoch, output_dir, n=8):
    """Generate sample textures and save them."""
    from torchvision.utils import save_image

    transformer.eval()
    vqvae.eval()

    prompts = ["iron sword", "diamond gem", "golden apple", "wooden shield",
               "magic wand", "red potion", "silver ring", "fire staff"]
    prompts = prompts[:n]

    tokens = torch.stack([dataset.encode_label(p) for p in prompts]).to(device)

    with torch.no_grad():
        indices = transformer.generate(tokens, temperature=0.8, top_k=50)
        indices_grid = indices.view(-1, 4, 4)
        images = vqvae.decode(indices_grid)

    images = (images + 1) / 2  # [-1,1] -> [0,1]
    save_image(images, os.path.join(output_dir, f"gen_epoch_{epoch:04d}.png"), nrow=4)
    transformer.train()


def main():
    parser = argparse.ArgumentParser(description="Train Transformer (Stage 2)")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to trained VQ-VAE checkpoint")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataloader, dataset = create_dataloader(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Dataset: {len(dataset)} textures, vocab: {dataset.vocab_size} words")

    # Load frozen VQ-VAE
    vqvae = VQVAE(
        in_channels=4, hidden_dim=128, embed_dim=256,
        num_embeddings=args.codebook_size,
    ).to(device)

    ckpt = torch.load(args.vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    print("VQ-VAE loaded and frozen.")

    # Transformer
    transformer = TokenTransformer(
        codebook_size=args.codebook_size,
        seq_len=16,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=dataset.vocab_size,
        text_max_len=16,
    ).to(device)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        transformer.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(os.path.join(args.output_dir, "logs_transformer"))

    for epoch in range(start_epoch, args.epochs):
        transformer.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            text_tokens = batch["tokens"].to(device)

            # Encode images to VQ indices
            with torch.no_grad():
                indices = vqvae.encode(images)  # (B, 4, 4)
            indices_flat = indices.view(indices.shape[0], -1)  # (B, 16)

            # Transformer forward
            logits = transformer(indices_flat, text_tokens)  # (B, 16, codebook_size)
            loss = F.cross_entropy(
                logits.reshape(-1, args.codebook_size),
                indices_flat.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / n_batches
        writer.add_scalar("loss/ce", avg_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        # Generate samples
        generate_samples(transformer, vqvae, dataset, device, epoch, args.output_dir)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                transformer, optimizer, epoch, avg_loss,
                os.path.join(args.checkpoint_dir, f"transformer_epoch_{epoch+1:04d}.pt"),
            )

    writer.close()
    print("Transformer training complete.")


if __name__ == "__main__":
    main()
