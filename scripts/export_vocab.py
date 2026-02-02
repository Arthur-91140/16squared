"""
Export vocabulary from dataset metadata for standalone inference.

Usage:
    python scripts/export_vocab.py --data_dir data/raw
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    args = parser.parse_args()

    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    words = set()
    for entry in metadata:
        words.update(entry["label"].split())

    word2idx = {"<pad>": 0, "<unk>": 1}
    for w in sorted(words):
        word2idx[w] = len(word2idx)

    vocab_path = os.path.join(args.data_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(word2idx, f, indent=2)

    print(f"Exported vocabulary: {len(word2idx)} words -> {vocab_path}")


if __name__ == "__main__":
    main()
