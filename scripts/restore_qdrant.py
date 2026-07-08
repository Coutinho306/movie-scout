"""Restore Qdrant collections from local snapshot files.

Usage:
    uv run python3 scripts/restore_qdrant.py [--url URL] [--snapshots-dir DIR]

Defaults:
    --url           http://localhost:6333  (keyless local Qdrant)
    --snapshots-dir data/snapshots/

The script expects exactly one snapshot file per collection in the directory:
    data/snapshots/tmdb_movies*.snapshot
    data/snapshots/tmdb_reviews*.snapshot

If multiple snapshots exist for a collection, the most-recently modified file
is used.

Recovery flow per collection:
    1. Upload the snapshot file via
       POST /collections/{name}/snapshots/upload?priority=snapshot
       (creates the collection from the snapshot, replacing any existing one).

After restore, point counts are verified and printed.

No API key is required for a local Qdrant instance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx


COLLECTIONS = ["tmdb_movies", "tmdb_reviews"]


def find_snapshot(snapshots_dir: Path, collection: str) -> Path:
    candidates = sorted(
        snapshots_dir.glob(f"{collection}*.snapshot"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No snapshot file found for '{collection}' in {snapshots_dir}.\n"
            f"Run: uv run python3 scripts/snapshot_qdrant.py"
        )
    if len(candidates) > 1:
        print(
            f"  [{collection}] multiple snapshots found; using newest: {candidates[0].name}",
            flush=True,
        )
    return candidates[0]


def restore_collection(
    base_url: str,
    collection: str,
    snapshot_path: Path,
    client: httpx.Client,
) -> int:
    """Upload snapshot and return restored point count."""
    size_mb = snapshot_path.stat().st_size / (1024 * 1024)
    print(
        f"  [{collection}] uploading {snapshot_path.name} ({size_mb:.1f} MB) ...",
        flush=True,
    )

    url = (
        f"{base_url}/collections/{collection}/snapshots/upload"
        "?priority=snapshot"
    )
    with snapshot_path.open("rb") as fh:
        resp = client.post(
            url,
            content=fh,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600.0,
        )

    if resp.status_code not in (200, 201):
        print(
            f"  [{collection}] upload failed: HTTP {resp.status_code}: {resp.text[:400]}",
            file=sys.stderr,
        )
        return -1

    result = resp.json()
    if not result.get("result", False):
        print(
            f"  [{collection}] unexpected response: {result}",
            file=sys.stderr,
        )
        return -1

    # Fetch point count
    info_resp = client.get(
        f"{base_url}/collections/{collection}", timeout=30.0
    )
    info_resp.raise_for_status()
    count: int = info_resp.json()["result"]["points_count"]
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore Qdrant collections from local snapshot files."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant base URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--snapshots-dir",
        default="data/snapshots",
        help="Directory containing .snapshot files (default: data/snapshots)",
    )
    args = parser.parse_args()

    base_url: str = args.url.rstrip("/")
    snapshots_dir = Path(args.snapshots_dir)

    if not snapshots_dir.is_dir():
        print(
            f"ERROR: snapshots directory not found: {snapshots_dir}\n"
            "Run: uv run python3 scripts/snapshot_qdrant.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify Qdrant reachable
    with httpx.Client() as probe:
        try:
            r = probe.get(f"{base_url}/healthz", timeout=10.0)
            r.raise_for_status()
        except Exception as exc:
            print(
                f"ERROR: Qdrant not reachable at {base_url}: {exc}", file=sys.stderr
            )
            print(
                "Start it first: docker compose up -d qdrant", file=sys.stderr
            )
            sys.exit(1)

    print(f"Qdrant reachable at {base_url}")
    print(f"Snapshots directory: {snapshots_dir.resolve()}\n")

    results: dict[str, int] = {}

    with httpx.Client() as client:
        for collection in COLLECTIONS:
            try:
                snapshot_path = find_snapshot(snapshots_dir, collection)
            except FileNotFoundError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)

            count = restore_collection(base_url, collection, snapshot_path, client)
            if count < 0:
                print(f"ERROR: restore failed for '{collection}'", file=sys.stderr)
                sys.exit(1)
            results[collection] = count
            print(f"  [{collection}] restored: {count:,} points")

    print("\nRestore complete:")
    for col, count in results.items():
        print(f"  {col}: {count:,} points")

    # Sanity check known counts
    expected = {"tmdb_movies": 15_503, "tmdb_reviews": 36_716}
    ok = True
    for col, exp in expected.items():
        got = results.get(col, 0)
        if got != exp:
            print(
                f"  WARNING: {col} count mismatch — expected {exp:,}, got {got:,}",
                file=sys.stderr,
            )
            ok = False
    if ok:
        print("\nPoint counts match expected (15,503 / 36,716). Restore verified.")


if __name__ == "__main__":
    main()
