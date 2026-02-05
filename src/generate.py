"""
Inference: Generate Minecraft item textures from text prompts.

Usage:
    python -m src.generate --prompt "iron sword" --vqvae_ckpt checkpoints/vqvae.pt --transformer_ckpt checkpoints/transformer.pt

    # Batch generation
    python -m src.generate --prompts_file prompts.txt --output_dir outputs/generated
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from .vqvae import VQVAE
from .transformer import TokenTransformer


def postprocess(image_tensor: torch.Tensor) -> Image.Image:
    """Convert model output tensor to clean PNG with transparent background.

    Args:
        image_tensor: (4, 16, 16) tensor in [-1, 1]

    Returns:
        PIL Image in RGBA mode
    """
    # Denormalize to [0, 255]
    img = ((image_tensor + 1) / 2 * 255).clamp(0, 255)
    img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (16, 16, 4)

    # Clean alpha channel: threshold to binary
    alpha = img[:, :, 3]
    alpha[alpha < 128] = 0
    alpha[alpha >= 128] = 255
    img[:, :, 3] = alpha

    # Quantize palette: reduce to ~12 colors per channel for pixel art look
    for c in range(3):
        channel = img[:, :, c].astype(np.float32)
        # Quantize to 16 levels
        channel = np.round(channel / 17) * 17
        img[:, :, c] = channel.clip(0, 255).astype(np.uint8)

    # Zero out RGB where fully transparent
    transparent = img[:, :, 3] == 0
    img[transparent, :3] = 0

    return Image.fromarray(img, "RGBA")


class TextureGenerator:
    """High-level API for generating Minecraft item textures."""

    def __init__(
        self,
        vqvae_ckpt: str,
        transformer_ckpt: str,
        vocab_path: str,
        device: str = "cuda",
        codebook_size: int = 512,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load vocab
        with open(vocab_path, "r") as f:
            self.word2idx = json.load(f)
        vocab_size = len(self.word2idx)

        # Load VQ-VAE
        self.vqvae = VQVAE(
            in_channels=4, hidden_dim=64, embed_dim=64,
            num_embeddings=codebook_size,
        ).to(self.device)
        ckpt = torch.load(vqvae_ckpt, map_location=self.device)
        self.vqvae.load_state_dict(ckpt["model_state_dict"])
        self.vqvae.eval()

        # Load Transformer
        self.transformer = TokenTransformer(
            codebook_size=codebook_size,
            seq_len=256,  # 16x16
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            text_max_len=16,
        ).to(self.device)
        ckpt = torch.load(transformer_ckpt, map_location=self.device)
        self.transformer.load_state_dict(ckpt["model_state_dict"])
        self.transformer.eval()

    def encode_prompt(self, prompt: str, max_len: int = 16) -> torch.Tensor:
        words = prompt.lower().strip().split()
        indices = [self.word2idx.get(w, self.word2idx.get("<unk>", 1)) for w in words]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Image.Image:
        """Generate a single 16x16 texture from a text prompt."""
        tokens = self.encode_prompt(prompt)
        indices = self.transformer.generate(tokens, temperature=temperature, top_k=top_k)
        indices_grid = indices.view(1, 16, 16)
        image = self.vqvae.decode(indices_grid)
        return postprocess(image[0])

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> list[Image.Image]:
        """Generate textures for multiple prompts."""
        tokens = torch.cat([self.encode_prompt(p) for p in prompts], dim=0)
        indices = self.transformer.generate(tokens, temperature=temperature, top_k=top_k)
        indices_grid = indices.view(-1, 16, 16)
        images = self.vqvae.decode(indices_grid)
        return [postprocess(images[i]) for i in range(len(prompts))]

    @torch.no_grad()
    def generate_variations(
        self,
        prompt: str,
        n: int = 4,
        temperature: float = 1.0,
        top_k: int = 100,
    ) -> list[Image.Image]:
        """Generate multiple variations for the same prompt."""
        return self.generate_batch([prompt] * n, temperature=temperature, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Generate Minecraft item textures")
    parser.add_argument("--prompt", type=str, help="Single text prompt")
    parser.add_argument("--prompts_file", type=str, help="File with one prompt per line")
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--transformer_ckpt", required=True)
    parser.add_argument("--vocab_path", default="data/raw/vocab.json")
    parser.add_argument("--output_dir", default="outputs/generated")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--variations", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generator = TextureGenerator(
        vqvae_ckpt=args.vqvae_ckpt,
        transformer_ckpt=args.transformer_ckpt,
        vocab_path=args.vocab_path,
    )

    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

    for prompt in prompts:
        safe_name = prompt.replace(" ", "_")
        if args.variations > 1:
            images = generator.generate_variations(prompt, n=args.variations)
            for i, img in enumerate(images):
                path = os.path.join(args.output_dir, f"{safe_name}_v{i}.png")
                img.save(path)
                print(f"Saved: {path}")
        else:
            img = generator.generate(prompt, temperature=args.temperature, top_k=args.top_k)
            path = os.path.join(args.output_dir, f"{safe_name}.png")
            img.save(path)
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
