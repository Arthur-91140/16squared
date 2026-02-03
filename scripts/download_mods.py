"""
Download Minecraft mod JARs from Modrinth API for texture extraction.

Searches for mods across multiple categories and downloads their JARs.
Downloads run in parallel with search for maximum speed.

Usage:
    python scripts/download_mods.py --output_dir dataset/mods --max_mods 15000 --workers 8
"""

import argparse
import os
import sys
import time
import json
import queue
import threading
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "https://api.modrinth.com/v2"
HEADERS = {
    "User-Agent": "16squared-texture-generator/1.0 (github.com/Arthur-91140/16squared)",
}

# Thread-safe counters
download_lock = threading.Lock()
stats = {"downloaded": 0, "skipped": 0, "failed": 0, "queued": 0}


def api_get(endpoint: str, params: dict = None) -> dict | list | None:
    url = f"{API_BASE}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def search_mods(query: str, offset: int = 0, limit: int = 100, facets: str = None, index: str = "downloads") -> list:
    """Search Modrinth for mods."""
    params = {
        "offset": str(offset),
        "limit": str(limit),
        "index": index,
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
    tmp_dest = dest + ".tmp"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(tmp_dest, "wb") as f:
                f.write(resp.read())
        os.rename(tmp_dest, dest)
        return True
    except Exception:
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)
        return False


def download_worker(project: dict, output_dir: str, idx: int, total: int) -> bool:
    """Download a single mod JAR. Called from thread pool."""
    slug = project["slug"]

    versions = get_project_versions(project["project_id"])
    if not versions:
        with download_lock:
            stats["failed"] += 1
            print(f"[{idx}/{total}] {slug}... SKIP (no versions)")
        return False

    # Try release versions first
    for version in versions:
        if version.get("version_type") != "release":
            continue
        for f in version.get("files", []):
            if f["filename"].endswith(".jar"):
                dest = os.path.join(output_dir, f["filename"])
                if os.path.exists(dest):
                    with download_lock:
                        stats["skipped"] += 1
                        print(f"[{idx}/{total}] {slug}... EXISTS")
                    return True
                if download_file(f["url"], dest):
                    with download_lock:
                        stats["downloaded"] += 1
                        print(f"[{idx}/{total}] {slug}... OK")
                    return True

    # Fallback: any version
    for version in versions[:3]:
        for f in version.get("files", []):
            if f["filename"].endswith(".jar"):
                dest = os.path.join(output_dir, f["filename"])
                if os.path.exists(dest):
                    with download_lock:
                        stats["skipped"] += 1
                        print(f"[{idx}/{total}] {slug}... EXISTS")
                    return True
                if download_file(f["url"], dest):
                    with download_lock:
                        stats["downloaded"] += 1
                        print(f"[{idx}/{total}] {slug}... OK")
                    return True

    with download_lock:
        stats["failed"] += 1
        print(f"[{idx}/{total}] {slug}... SKIP (no JAR)")
    return False


def search_and_download(output_dir: str, max_mods: int, workers: int):
    """Search for mods and download in parallel."""
    projects = {}
    projects_lock = threading.Lock()
    download_queue = queue.Queue()
    search_done = threading.Event()

    def _add_project(hit):
        pid = hit["project_id"]
        with projects_lock:
            if pid not in projects:
                projects[pid] = {
                    "project_id": pid,
                    "slug": hit.get("slug", pid),
                    "title": hit.get("title", ""),
                    "downloads": hit.get("downloads", 0),
                }
                download_queue.put(projects[pid])
                return True
        return False

    def _exhaust_search(query: str, facets: str, index: str = "downloads"):
        """Paginate through all results for a query+facets combo."""
        offset = 0
        while offset < 10000:
            with projects_lock:
                if len(projects) >= max_mods:
                    return
            hits = search_mods(query, offset=offset, limit=100, facets=facets, index=index)
            if not hits:
                break
            for hit in hits:
                _add_project(hit)
            if len(hits) < 100:
                break
            offset += 100
            time.sleep(0.1)

    def search_thread():
        """Thread that performs all searches."""
        print(f"Searching Modrinth for mods (target: {max_mods})...")

        # Strategy 1: Different sort orders
        for sort_index in ["downloads", "updated", "newest", "follows"]:
            with projects_lock:
                if len(projects) >= max_mods:
                    break
            _exhaust_search("", '[["project_type:mod"]]', index=sort_index)
            with projects_lock:
                print(f"  [sort={sort_index}]: {len(projects)} unique projects")

        # Strategy 2: Loaders
        loaders = ["forge", "fabric", "quilt", "neoforge"]
        for loader in loaders:
            with projects_lock:
                if len(projects) >= max_mods:
                    break
            _exhaust_search("", f'[["project_type:mod"],["categories:{loader}"]]')
            with projects_lock:
                print(f"  [loader={loader}]: {len(projects)} unique projects")

        # Strategy 3: Minecraft versions
        versions = [
            "1.21.4", "1.21.1", "1.20.4", "1.20.1", "1.19.2", "1.18.2",
            "1.17.1", "1.16.5", "1.15.2", "1.14.4", "1.12.2", "1.10.2",
            "1.8.9", "1.7.10",
        ]
        for ver in versions:
            with projects_lock:
                if len(projects) >= max_mods:
                    break
            _exhaust_search("", f'[["project_type:mod"],["versions:{ver}"]]')
            with projects_lock:
                print(f"  [version={ver}]: {len(projects)} unique projects")

        # Strategy 4: Letter searches
        for letter in "abcdefghijklmnopqrstuvwxyz0123456789":
            with projects_lock:
                if len(projects) >= max_mods:
                    break
            _exhaust_search(letter, '[["project_type:mod"]]')

        with projects_lock:
            print(f"\nSearch complete: {len(projects)} unique projects found")
        search_done.set()

    def download_thread(executor, output_dir):
        """Thread that submits download tasks from queue."""
        futures = []
        idx = 0

        while True:
            try:
                project = download_queue.get(timeout=1.0)
                idx += 1
                with projects_lock:
                    total = len(projects)
                future = executor.submit(download_worker, project, output_dir, idx, total)
                futures.append(future)
            except queue.Empty:
                if search_done.is_set() and download_queue.empty():
                    break

        # Wait for remaining downloads
        for future in as_completed(futures):
            pass

    # Start search thread
    searcher = threading.Thread(target=search_thread, daemon=True)
    searcher.start()

    # Start download pool
    print(f"Starting {workers} download workers...\n")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloader = threading.Thread(target=download_thread, args=(executor, output_dir), daemon=True)
        downloader.start()

        # Wait for both to complete
        searcher.join()
        downloader.join()

    return projects


def main():
    parser = argparse.ArgumentParser(description="Download Minecraft mods from Modrinth")
    parser.add_argument("--output_dir", default="dataset/mods")
    parser.add_argument("--max_mods", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    projects = search_and_download(args.output_dir, args.max_mods, args.workers)

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(list(projects.values()), f, indent=2)

    print(f"\nDone!")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Already existed: {stats['skipped']}")
    print(f"  Failed/Skipped: {stats['failed']}")
    print(f"  JARs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
