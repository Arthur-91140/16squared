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

    # Track filename usage for conflict resolution
    name_counts = defaultdict(int)
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
    conflicts = 0

    for png_path in all_pngs:
        mod_name = png_path.parent.name
        original_name = png_path.name
        base_name = png_path.stem
        ext = png_path.suffix

        # Determine output filename
        if name_counts[original_name] == 0:
            # First occurrence, use original name
            out_name = original_name
        else:
            # Conflict, add numeric suffix
            out_name = f"{base_name}_{name_counts[original_name]}{ext}"
            conflicts += 1

        name_counts[original_name] += 1

        # Copy file
        out_path = os.path.join(args.output_dir, out_name)
        shutil.copy2(png_path, out_path)
        copied += 1

        # Build metadata entry
        rel_key = f"{mod_name}/{original_name}"
        if rel_key in input_meta:
            entry = input_meta[rel_key].copy()
            entry["filename"] = out_name
        else:
            label = base_name.replace("_", " ").replace("-", " ").lower()
            entry = {
                "filename": out_name,
                "label": label,
                "source_jar": mod_name,
                "original_name": original_name,
            }

        metadata.append(entry)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  Textures copied: {copied}")
    print(f"  Naming conflicts resolved: {conflicts}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
