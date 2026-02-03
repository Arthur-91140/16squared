"""
Extract 16x16 item textures from downloaded mod JARs and delete JARs after extraction.

Scans each JAR for assets/*/textures/item(s)/*.png, keeps only 16x16 PNGs,
then deletes the JAR to save disk space.

Usage:
    python scripts/extract_and_cleanup.py --mods_dir dataset/mods --output_dir data/raw --workers 8
"""

import argparse
import os
import json
import zipfile
import threading
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# Thread-safe
lock = threading.Lock()
stats = {"extracted": 0, "skipped": 0, "jars_processed": 0, "jars_failed": 0}
metadata = []


def is_valid_16x16(img: Image.Image) -> bool:
    """Check if image is exactly 16x16 with some visible pixels."""
    if img.size != (16, 16):
        return False
    if img.mode != "RGBA":
        try:
            img = img.convert("RGBA")
        except Exception:
            return False
    # Check not fully transparent
    alpha = img.getchannel("A")
    if alpha.getextrema()[1] == 0:
        return False
    return True


def extract_from_jar(jar_path: str, output_dir: str) -> tuple[int, int]:
    """Extract 16x16 item textures from a JAR file.

    Returns (extracted_count, skipped_count)
    """
    extracted = 0
    skipped = 0
    local_meta = []

    try:
        zf = zipfile.ZipFile(jar_path, "r")
    except (zipfile.BadZipFile, Exception):
        return 0, 0

    jar_name = Path(jar_path).stem

    for entry in zf.namelist():
        # Match assets/*/textures/item/*.png or assets/*/textures/items/*.png
        parts = entry.lower().split("/")
        if len(parts) < 5:
            continue
        if parts[0] != "assets":
            continue
        if parts[2] != "textures":
            continue
        if parts[3] not in ("item", "items"):
            continue
        if not entry.lower().endswith(".png"):
            continue
        # Skip subdirectories (armor layers, etc.)
        if len(parts) > 5:
            continue

        try:
            data = zf.read(entry)
            img = Image.open(BytesIO(data))

            # Handle animated textures (spritesheet with height > width)
            if img.size[1] > img.size[0] and img.size[0] == 16:
                img = img.crop((0, 0, 16, 16))

            if not is_valid_16x16(img):
                skipped += 1
                continue

            img = img.convert("RGBA")

            # Generate unique filename, organized by mod
            namespace = parts[1]
            tex_name = Path(parts[-1]).stem

            # Create mod subdirectory
            mod_dir = os.path.join(output_dir, jar_name)
            os.makedirs(mod_dir, exist_ok=True)

            safe_name = f"{namespace}__{tex_name}.png"
            out_path = os.path.join(mod_dir, safe_name)

            # Skip if already exists
            if os.path.exists(out_path):
                skipped += 1
                continue

            img.save(out_path, "PNG")
            extracted += 1

            # Build label from filename
            label = tex_name.replace("_", " ").replace("-", " ").lower()
            local_meta.append({
                "filename": f"{jar_name}/{safe_name}",
                "label": label,
                "source_jar": jar_name,
                "namespace": namespace,
            })

        except Exception:
            skipped += 1
            continue

    zf.close()

    # Add to global metadata
    with lock:
        metadata.extend(local_meta)

    return extracted, skipped


def process_jar(jar_path: str, output_dir: str, delete_after: bool, idx: int, total: int) -> bool:
    """Process a single JAR: extract textures, optionally delete."""
    jar_name = Path(jar_path).name

    extracted, skipped = extract_from_jar(jar_path, output_dir)

    with lock:
        if extracted > 0:
            stats["extracted"] += extracted
            stats["skipped"] += skipped
            stats["jars_processed"] += 1
            print(f"[{idx}/{total}] {jar_name}: {extracted} textures extracted")
        else:
            stats["jars_failed"] += 1
            print(f"[{idx}/{total}] {jar_name}: no valid textures")

    # Delete JAR after extraction
    if delete_after:
        try:
            os.remove(jar_path)
        except Exception:
            pass

    return extracted > 0


def main():
    parser = argparse.ArgumentParser(description="Extract 16x16 textures from mod JARs")
    parser.add_argument("--mods_dir", default="dataset/mods", help="Directory containing JAR files")
    parser.add_argument("--output_dir", default="dataset/mod-filtered-png", help="Output directory for textures")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--keep_jars", action="store_true", help="Don't delete JARs after extraction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all JARs
    jars = list(Path(args.mods_dir).glob("*.jar"))
    print(f"Found {len(jars)} JAR files in {args.mods_dir}")

    if not jars:
        print("No JARs found. Exiting.")
        return

    delete_after = not args.keep_jars
    if delete_after:
        print("JARs will be DELETED after extraction. Use --keep_jars to prevent this.\n")
    else:
        print("JARs will be kept after extraction.\n")

    # Process in parallel
    total = len(jars)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_jar, str(jar), args.output_dir, delete_after, i+1, total): jar
            for i, jar in enumerate(jars)
        }
        for future in as_completed(futures):
            pass

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  JARs processed: {stats['jars_processed']}")
    print(f"  JARs with no textures: {stats['jars_failed']}")
    print(f"  Textures extracted: {stats['extracted']}")
    print(f"  Textures skipped: {stats['skipped']}")
    print(f"  Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
