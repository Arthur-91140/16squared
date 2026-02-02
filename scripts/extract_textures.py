"""
Extract item textures from Minecraft mod JARs.

Usage:
    python scripts/extract_textures.py --input_dir /path/to/mods --output_dir data/raw

Each JAR is a ZIP containing textures at:
    assets/<modid>/textures/item/*.png
    assets/<modid>/textures/items/*.png
"""

import argparse
import json
import os
import re
import zipfile
from pathlib import Path

from PIL import Image


def sanitize_label(mod_id: str, filename: str) -> str:
    """Convert a texture filename into a human-readable label.

    Examples:
        'diamond_sword.png' -> 'diamond sword'
        'goldenApple.png' -> 'golden apple'
    """
    name = Path(filename).stem
    # camelCase -> snake_case
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    # Replace separators with spaces
    name = name.replace("_", " ").replace("-", " ").lower().strip()
    return name


def is_valid_item_texture(img: Image.Image) -> bool:
    """Check if image is a valid 16x16 RGBA item texture (not animated)."""
    if img.size != (16, 16):
        return False
    if img.mode != "RGBA":
        try:
            img = img.convert("RGBA")
        except Exception:
            return False
    # Check it's not fully transparent
    alpha = img.getchannel("A")
    if alpha.getextrema()[1] == 0:
        return False
    return True


def extract_from_jar(jar_path: str, output_dir: str, metadata: list):
    """Extract item textures from a single JAR file."""
    try:
        zf = zipfile.ZipFile(jar_path, "r")
    except (zipfile.BadZipFile, Exception):
        return

    mod_id = Path(jar_path).stem
    item_patterns = [
        re.compile(r"^assets/([^/]+)/textures/items?/(.+\.png)$", re.IGNORECASE),
    ]

    for entry in zf.namelist():
        for pattern in item_patterns:
            match = pattern.match(entry)
            if not match:
                continue

            namespace = match.group(1)
            tex_filename = match.group(2)

            # Skip subdirectories (armor layers, etc.)
            if "/" in tex_filename:
                continue

            try:
                data = zf.read(entry)
                from io import BytesIO
                img = Image.open(BytesIO(data))

                # Handle animated textures (height > width = spritesheet)
                if img.size[1] > img.size[0] and img.size[0] == 16:
                    img = img.crop((0, 0, 16, 16))
                elif img.size != (16, 16):
                    continue

                img = img.convert("RGBA")

                if not is_valid_item_texture(img):
                    continue

                label = sanitize_label(namespace, tex_filename)
                safe_name = f"{namespace}__{Path(tex_filename).stem}.png"
                out_path = os.path.join(output_dir, safe_name)

                img.save(out_path, "PNG")
                metadata.append({
                    "filename": safe_name,
                    "label": label,
                    "source_mod": mod_id,
                    "namespace": namespace,
                })
            except Exception:
                continue

    zf.close()


def main():
    parser = argparse.ArgumentParser(description="Extract Minecraft item textures from mod JARs")
    parser.add_argument("--input_dir", required=True, help="Directory containing .jar files")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory for textures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = []

    jars = list(Path(args.input_dir).glob("*.jar"))
    print(f"Found {len(jars)} JAR files")

    for i, jar_path in enumerate(jars):
        print(f"[{i+1}/{len(jars)}] Processing {jar_path.name}...")
        extract_from_jar(str(jar_path), args.output_dir, metadata)

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Extracted {len(metadata)} textures to {args.output_dir}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
