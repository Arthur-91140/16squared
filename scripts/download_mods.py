"""
Download Minecraft mod JARs from Modrinth API for texture extraction.

Searches for mods across multiple categories and downloads their JARs.
Targets mods with item textures (16x16).

Usage:
    python scripts/download_mods.py --output_dir dataset/mods --max_mods 5000
"""

import argparse
import os
import sys
import time
import json
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

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
        "offset": str(offset),
        "limit": str(limit),
        "index": "downloads",
        "facets": facets or '[["project_type:mod"]]',
    }
    if query:
        params["query"] = query
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
    """Collect project IDs by searching across many queries and facet combos."""
    projects = {}

    def _add_hits(hits):
        for hit in hits:
            pid = hit["project_id"]
            if pid not in projects:
                projects[pid] = {
                    "project_id": pid,
                    "slug": hit.get("slug", pid),
                    "title": hit.get("title", ""),
                    "downloads": hit.get("downloads", 0),
                }

    def _exhaust_search(query: str, facets: str, label: str):
        """Paginate through all results for a query+facets combo (up to 10000)."""
        offset = 0
        while offset < 10000:
            hits = search_mods(query, offset=offset, limit=100, facets=facets)
            if not hits:
                break
            _add_hits(hits)
            if len(hits) < 100:
                break
            offset += 100
            time.sleep(0.25)
        print(f"  [{label}]: {len(projects)} unique projects so far")

    print(f"Searching Modrinth for mods (target: {max_mods})...")

    # Strategy 1: Browse by sort order with no query (different sort = different results)
    for sort_index in ["downloads", "updated", "newest", "follows", "relevance"]:
        facets = '[["project_type:mod"]]'
        offset = 0
        while offset < 10000:
            params = {
                "offset": str(offset),
                "limit": "100",
                "index": sort_index,
                "facets": facets,
            }
            data = api_get("/search", params)
            if not data or "hits" not in data or not data["hits"]:
                break
            _add_hits(data["hits"])
            if len(data["hits"]) < 100:
                break
            offset += 100
            time.sleep(0.25)
        print(f"  [sort={sort_index}]: {len(projects)} unique projects so far")

    # Strategy 2: Browse by loader facets
    loaders = ["forge", "fabric", "quilt", "neoforge", "rift", "liteloader"]
    for loader in loaders:
        if len(projects) >= max_mods:
            break
        facets = f'[["project_type:mod"],["categories:{loader}"]]'
        _exhaust_search("", facets, f"loader={loader}")

    # Strategy 3: Browse by Minecraft version facets
    versions = [
        "1.21.4", "1.21.3", "1.21.2", "1.21.1", "1.21",
        "1.20.6", "1.20.4", "1.20.2", "1.20.1", "1.20",
        "1.19.4", "1.19.2", "1.19",
        "1.18.2", "1.18.1", "1.18",
        "1.17.1", "1.16.5", "1.16.4", "1.16.3",
        "1.15.2", "1.14.4", "1.12.2", "1.12.1",
        "1.11.2", "1.10.2", "1.9.4", "1.8.9", "1.7.10",
    ]
    for ver in versions:
        if len(projects) >= max_mods:
            break
        facets = f'[["project_type:mod"],["versions:{ver}"]]'
        _exhaust_search("", facets, f"version={ver}")

    # Strategy 4: Keyword searches for remaining coverage
    search_terms = [
        "sword", "armor", "tool", "food", "ore", "gem", "magic",
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
        "hammer", "spear", "dagger", "rapier", "scythe", "trident",
        "arrow", "bolt", "bullet", "gun", "rifle", "cannon",
        "spell", "ritual", "altar", "totem", "talisman",
        "elixir", "brew", "flask", "vial", "bottle",
        "cape", "cloak", "boots", "gloves", "leggings", "chestplate",
        "pendant", "bracelet", "earring", "crown", "tiara",
        "apple", "bread", "cake", "pie", "stew", "soup", "meat",
        "wool", "leather", "silk", "cloth", "fabric", "string",
        "brick", "stone", "wood", "log", "plank", "slab",
        "rail", "minecart", "boat", "elytra", "saddle",
        "compass", "map", "clock", "spyglass", "telescope",
        "redstone", "piston", "hopper", "dispenser", "dropper",
        "enchant", "anvil", "grindstone", "smithing",
        "spawn", "egg", "bucket", "shears", "flint",
        "pearl", "blaze", "ghast", "skeleton", "zombie", "creeper",
        "witch", "villager", "pillager", "phantom", "enderman",
        "slime", "honey", "wax", "dye", "ink", "paint",
        "music", "disc", "note", "horn", "bell", "drum",
    ]
    for term in search_terms:
        if len(projects) >= max_mods:
            break
        facets = '[["project_type:mod"]]'
        _exhaust_search(term, facets, f"query={term}")

    # Strategy 5: Single letter searches for broad coverage
    for letter in "abcdefghijklmnopqrstuvwxyz":
        if len(projects) >= max_mods:
            break
        facets = '[["project_type:mod"]]'
        _exhaust_search(letter, facets, f"letter={letter}")

    print(f"\nTotal unique projects found: {len(projects)}")
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
