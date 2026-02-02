"""
Prepare and clean the extracted dataset.

Steps:
1. Remove duplicates (by pixel content hash)
2. Filter out bad textures (fully opaque backgrounds, too few non-transparent pixels)
3. Validate all images are 16x16 RGBA
4. Generate train/val split
5. Print dataset statistics

Usage:
    python scripts/prepare_dataset.py --data_dir data/raw --output_dir data/processed
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter

import numpy as np
from PIL import Image


def image_hash(img: Image.Image) -> str:
    return hashlib.md5(np.array(img).tobytes()).hexdigest()


def is_good_texture(img: Image.Image, min_opaque_pixels: int = 10, max_opaque_ratio: float = 0.95) -> bool:
    """Filter out bad textures."""
    arr = np.array(img)
    alpha = arr[:, :, 3]

    opaque_count = np.sum(alpha > 128)

    # Too few visible pixels — probably broken
    if opaque_count < min_opaque_pixels:
        return False

    # Almost fully opaque — likely a block texture, not an item
    if opaque_count / (16 * 16) > max_opaque_ratio:
        return False

    # Check it's not a single solid color
    rgb_opaque = arr[alpha > 128, :3]
    unique_colors = len(np.unique(rgb_opaque.reshape(-1, 3), axis=0))
    if unique_colors < 3:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)

    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print(f"Raw dataset: {len(metadata)} entries")

    # Deduplicate and filter
    seen_hashes = set()
    clean = []
    stats = Counter()

    for entry in metadata:
        img_path = os.path.join(args.data_dir, entry["filename"])
        if not os.path.exists(img_path):
            stats["missing"] += 1
            continue

        img = Image.open(img_path).convert("RGBA")

        if img.size != (16, 16):
            stats["wrong_size"] += 1
            continue

        h = image_hash(img)
        if h in seen_hashes:
            stats["duplicate"] += 1
            continue
        seen_hashes.add(h)

        if not is_good_texture(img):
            stats["bad_quality"] += 1
            continue

        clean.append(entry)
        stats["kept"] += 1

    print(f"Filtering stats: {dict(stats)}")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(clean)

    val_count = max(1, int(len(clean) * args.val_ratio))
    val_entries = clean[:val_count]
    train_entries = clean[val_count:]

    # Copy files and save metadata
    for split_name, entries in [("train", train_entries), ("val", val_entries)]:
        split_dir = os.path.join(args.output_dir, split_name)
        split_meta = []

        for entry in entries:
            src = os.path.join(args.data_dir, entry["filename"])
            dst = os.path.join(split_dir, entry["filename"])
            shutil.copy2(src, dst)
            split_meta.append(entry)

        with open(os.path.join(split_dir, "metadata.json"), "w") as f:
            json.dump(split_meta, f, indent=2)

        print(f"{split_name}: {len(entries)} textures")

    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
