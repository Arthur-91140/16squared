"""
Flatten extracted textures into a single training directory.

Takes the mod-organized output from extract_and_cleanup.py and puts all
textures in one flat folder. Conflicts are resolved with numeric suffixes.

Usage:
    python scripts/flatten_dataset.py --input_dir dataset/mod-filtered-png --output_dir data/raw
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Flatten textures into training directory")
    parser.add_argument("--input_dir", default="dataset/mod-filtered-png", help="Input directory with mod subfolders")
    parser.add_argument("--output_dir", default="data/raw", help="Output flat directory for training")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Track seen filenames to skip duplicates
    seen_names = set()
    metadata = []

    # Load existing metadata if present
    input_meta_path = os.path.join(args.input_dir, "metadata.json")
    input_meta = {}
    if os.path.exists(input_meta_path):
        with open(input_meta_path, "r") as f:
            for entry in json.load(f):
                input_meta[entry["filename"]] = entry

    # Find all PNGs in subdirectories
    input_path = Path(args.input_dir)
    all_pngs = list(input_path.glob("*/*.png"))
    print(f"Found {len(all_pngs)} textures in {args.input_dir}")

    copied = 0
    skipped = 0

    for png_path in all_pngs:
        mod_name = png_path.parent.name
        original_name = png_path.name
        base_name = png_path.stem

        # Skip duplicates
        if original_name in seen_names:
            skipped += 1
            continue

        seen_names.add(original_name)

        # Copy file
        out_path = os.path.join(args.output_dir, original_name)
        shutil.copy2(png_path, out_path)
        copied += 1

        # Build metadata entry
        rel_key = f"{mod_name}/{original_name}"
        if rel_key in input_meta:
            entry = input_meta[rel_key].copy()
            entry["filename"] = original_name
        else:
            label = base_name.replace("_", " ").replace("-", " ").lower()
            entry = {
                "filename": original_name,
                "label": label,
                "source_jar": mod_name,
            }

        metadata.append(entry)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  Textures copied: {copied}")
    print(f"  Duplicates skipped: {skipped}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
