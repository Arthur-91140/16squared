"""
Dataset and DataLoader for 16x16 Minecraft item textures.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MinecraftItemDataset(Dataset):
    """Dataset of 16x16 RGBA Minecraft item textures with text labels."""

    def __init__(self, data_dir: str, tokenizer=None, max_token_len: int = 16):
        self.data_dir = data_dir
        self.max_token_len = max_token_len
        self.tokenizer = tokenizer

        meta_path = os.path.join(data_dir, "metadata.json")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        # Build vocabulary from labels if no tokenizer provided
        if self.tokenizer is None:
            self._build_vocab()

    def _build_vocab(self):
        """Build a simple word-level vocabulary from all labels."""
        words = set()
        for entry in self.metadata:
            words.update(entry["label"].split())
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        for w in sorted(words):
            self.word2idx[w] = len(self.word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode_label(self, label: str) -> torch.Tensor:
        """Encode a text label into token indices."""
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                label,
                max_length=self.max_token_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return tokens["input_ids"].squeeze(0)

        words = label.split()
        indices = [self.word2idx.get(w, 1) for w in words]
        # Pad or truncate
        if len(indices) < self.max_token_len:
            indices += [0] * (self.max_token_len - len(indices))
        else:
            indices = indices[: self.max_token_len]
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        img_path = os.path.join(self.data_dir, entry["filename"])

        try:
            img = Image.open(img_path).convert("RGBA")
            # Ensure 16x16
            if img.size != (16, 16):
                img = img.resize((16, 16), Image.NEAREST)
        except Exception:
            # Return blank image on error
            img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))

        # Normalize to [-1, 1]
        pixels = np.array(img, dtype=np.float32) / 127.5 - 1.0  # (16, 16, 4)
        pixels = torch.from_numpy(pixels).permute(2, 0, 1)  # (4, 16, 16)

        tokens = self.encode_label(entry["label"])
        label_text = entry["label"]

        return {
            "image": pixels,
            "tokens": tokens,
            "label": label_text,
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    tokenizer=None,
) -> tuple[DataLoader, MinecraftItemDataset]:
    """Create a DataLoader for the Minecraft item dataset."""
    dataset = MinecraftItemDataset(data_dir, tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset
