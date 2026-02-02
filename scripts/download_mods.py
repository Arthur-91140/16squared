"""
Download Minecraft mod JARs from Modrinth API for texture extraction.

Searches for mods across multiple categories and downloads their JARs.
Targets mods with item textures (16x16).

Usage:
    python scripts/download_mods.py --output_dir dataset/mods --max_mods 5000
"""

import argparse
import os
import time
import json
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "https://api.modrinth.com/v2"
HEADERS = {
    "User-Agent": "16squared-texture-generator/1.0 (github.com/Arthur-91140/16squared)",
}


def api_get(endpoint: str, params: dict = None) -> dict | list | None:
    url = f"{API_BASE}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        print(f"  API error: {e}")
        return None


def search_mods(query: str, offset: int = 0, limit: int = 100, facets: str = None) -> list:
    """Search Modrinth for mods."""
    params = {
        "query": query,
        "offset": offset,
        "limit": limit,
        "index": "downloads",
        "facets": facets or '[["project_type:mod"]]',
    }
    data = api_get("/search", params)
    if data and "hits" in data:
        return data["hits"]
    return []


def get_project_versions(project_id: str) -> list:
    """Get all versions for a project."""
    data = api_get(f"/project/{project_id}/version")
    return data if data else []


def download_file(url: str, dest: str) -> bool:
    """Download a file."""
    if os.path.exists(dest):
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(dest, "wb") as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def collect_project_ids(max_mods: int) -> list[dict]:
    """Collect project IDs by searching across many queries."""
    projects = {}

    # Generic searches to find lots of mods
    search_terms = [
        "", "sword", "armor", "tool", "food", "ore", "gem", "magic",
        "potion", "weapon", "ring", "staff", "bow", "shield", "helmet",
        "pickaxe", "axe", "wand", "crystal", "ingot", "dust", "rod",
        "amulet", "charm", "rune", "scroll", "book", "crop", "seed",
        "berry", "fruit", "fish", "mob", "dragon", "golem", "boss",
        "tech", "machine", "pipe", "wire", "gear", "motor", "energy",
        "furnace", "chest", "bag", "backpack", "key", "coin", "medal",
        "biome", "dimension", "block", "decoration", "furniture",
        "light", "lamp", "torch", "lantern", "candle",
        "storage", "inventory", "crafting", "smelting",
        "adventure", "exploration", "dungeon", "loot",
        "nature", "animal", "plant", "flower", "tree",
        "nether", "end", "void", "sky", "ocean", "cave",
        "copper", "iron", "gold", "diamond", "emerald", "netherite",
        "ruby", "sapphire", "amethyst", "obsidian", "quartz",
    ]

    print(f"Searching Modrinth for mods (target: {max_mods})...")

    for term in search_terms:
        if len(projects) >= max_mods:
            break

        offset = 0
        while offset < 500 and len(projects) < max_mods:
            hits = search_mods(term, offset=offset, limit=100)
            if not hits:
                break

            for hit in hits:
                pid = hit["project_id"]
                if pid not in projects:
                    projects[pid] = {
                        "project_id": pid,
                        "slug": hit.get("slug", pid),
                        "title": hit.get("title", ""),
                        "downloads": hit.get("downloads", 0),
                    }

            offset += 100
            time.sleep(0.3)  # Be nice to the API

        print(f"  '{term}': {len(projects)} unique projects so far")

    return list(projects.values())


def download_mod_jar(project: dict, output_dir: str) -> bool:
    """Download the latest JAR for a project."""
    versions = get_project_versions(project["project_id"])
    if not versions:
        return False

    # Pick the latest release version, prefer Forge/Fabric for more item mods
    for version in versions:
        if version.get("version_type") != "release":
            continue
        files = version.get("files", [])
        for f in files:
            if f["filename"].endswith(".jar"):
                dest = os.path.join(output_dir, f["filename"])
                if os.path.exists(dest):
                    return True
                return download_file(f["url"], dest)

    # Fallback: any version with a JAR
    for version in versions[:3]:
        files = version.get("files", [])
        for f in files:
            if f["filename"].endswith(".jar"):
                dest = os.path.join(output_dir, f["filename"])
                if os.path.exists(dest):
                    return True
                return download_file(f["url"], dest)

    return False


def main():
    parser = argparse.ArgumentParser(description="Download Minecraft mods from Modrinth")
    parser.add_argument("--output_dir", default="dataset/mods")
    parser.add_argument("--max_mods", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: Collect project IDs
    projects = collect_project_ids(args.max_mods)
    print(f"\nFound {len(projects)} unique projects. Starting downloads...\n")

    # Save project list for reference
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(projects, f, indent=2)

    # Phase 2: Download JARs
    downloaded = 0
    failed = 0

    for i, project in enumerate(projects):
        slug = project["slug"]
        print(f"[{i+1}/{len(projects)}] {slug}...", end=" ", flush=True)

        success = download_mod_jar(project, args.output_dir)
        if success:
            downloaded += 1
            print("OK")
        else:
            failed += 1
            print("SKIP")

        time.sleep(0.3)

    print(f"\nDone. Downloaded: {downloaded}, Failed/Skipped: {failed}")
    print(f"JARs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
