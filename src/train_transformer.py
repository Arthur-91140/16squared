"""
Stage 2: Train autoregressive Transformer conditioned on text.

Requires a trained VQ-VAE checkpoint from Stage 1.
The Transformer learns to generate VQ token sequences from text prompts.

Optimizations:
- TF32 for faster matmuls (2-3x speedup)
- BF16 mixed precision (more stable than FP16)
- torch.compile() with default mode
- Gradient accumulation for larger effective batch
- Learning rate warmup

Usage:
    python -m src.train_transformer --data_dir data/raw --vqvae_ckpt checkpoints/vqvae_epoch_0100.pt
"""

import argparse
import gc
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import create_dataloader
from .vqvae import VQVAE
from .transformer import TokenTransformer


def save_checkpoint(model, optimizer, epoch, loss, path):
    # Handle compiled model
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_samples(transformer, vqvae, dataset, device, epoch, output_dir, n=8):
    """Generate sample textures and save them."""
    from torchvision.utils import save_image
    import random

    transformer.eval()
    vqvae.eval()

    # Use random labels from the actual dataset vocab
    all_labels = [entry["label"] for entry in dataset.metadata]
    prompts = random.sample(all_labels, min(n, len(all_labels)))

    tokens = torch.stack([dataset.encode_label(p) for p in prompts]).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        indices = transformer.generate(tokens, temperature=0.8, top_k=50)
        indices_grid = indices.view(-1, 16, 16)
        images = vqvae.decode(indices_grid)

        images = (images + 1) / 2  # [-1,1] -> [0,1]
        save_image(images, os.path.join(output_dir, f"gen_epoch_{epoch:04d}.png"), nrow=4)

    # Log prompts used
    with open(os.path.join(output_dir, f"gen_epoch_{epoch:04d}_prompts.txt"), "w") as f:
        for p in prompts:
            f.write(p + "\n")

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
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--sample_every", type=int, default=5, help="Save sample images every N epochs")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Learning rate warmup epochs")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable TF32 for faster matmuls on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 enabled for faster matmuls")

    # Data with persistent workers for faster loading
    dataloader, dataset = create_dataloader(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Dataset: {len(dataset)} textures, vocab: {dataset.vocab_size} words")
    print(f"Effective batch size: {args.batch_size * args.grad_accum} (batch={args.batch_size} x accum={args.grad_accum})")

    # Save vocab for inference
    vocab_path = os.path.join(args.checkpoint_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(dataset.word2idx, f, indent=2)
    print(f"Vocab saved to {vocab_path}")

    # Load frozen VQ-VAE (must match train_vqvae.py defaults: hidden_dim=128, embed_dim=256)
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
        seq_len=256,  # 16x16
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=dataset.vocab_size,
        text_max_len=16,
    ).to(device)

    # Compile for faster training
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling transformer with torch.compile()...")
        transformer = torch.compile(transformer, mode="default")

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr, fused=True)

    # Cosine scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model_to_load = transformer._orig_mod if hasattr(transformer, '_orig_mod') else transformer
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(os.path.join(args.output_dir, "logs_transformer"))

    for epoch in range(start_epoch, args.epochs):
        transformer.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            text_tokens = batch["tokens"].to(device, non_blocking=True)

            # Encode images to VQ indices
            with torch.no_grad():
                indices = vqvae.encode(images)
            indices_flat = indices.view(indices.shape[0], -1)

            # BF16 forward pass (no GradScaler needed)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = transformer(indices_flat, text_tokens)
                loss = F.cross_entropy(
                    logits.reshape(-1, args.codebook_size),
                    indices_flat.reshape(-1),
                )
                loss = loss / args.grad_accum  # Scale for accumulation

            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item() * args.grad_accum  # Unscale for logging

            # NaN/Inf detection
            if not (loss_val == loss_val) or loss_val > 1e6:
                print(f"\nNaN/Inf detected at batch! loss={loss_val}")
                print("Try: lower learning rate (--lr 5e-5)")
                return

            total_loss += loss_val
            n_batches += 1

            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg_loss = total_loss / n_batches
        writer.add_scalar("loss/ce", avg_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # Clear memory
        clear_memory()

        # Generate samples (less frequently)
        if (epoch + 1) % args.sample_every == 0:
            generate_samples(transformer, vqvae, dataset, device, epoch, args.output_dir)
            clear_memory()

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                transformer, optimizer, epoch, avg_loss,
                os.path.join(args.checkpoint_dir, f"transformer_epoch_{epoch+1:04d}.pt"),
            )

    writer.close()
    print("Transformer training complete.")


if __name__ == "__main__":
    main()
