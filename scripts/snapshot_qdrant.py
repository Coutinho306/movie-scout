"""Create Qdrant snapshots of both collections and save them to data/snapshots/.

Usage:
    uv run python3 scripts/snapshot_qdrant.py [--url URL] [--out-dir DIR]

Defaults:
    --url      http://localhost:6333  (keyless local Qdrant)
    --out-dir  data/snapshots/

The script calls the Qdrant REST snapshot API:
    POST /collections/{name}/snapshots   -> triggers creation on the server
    GET  /collections/{name}/snapshots   -> lists snapshots to get the filename
    GET  /collections/{name}/snapshots/{snapshot_name} -> downloads the file

No API key is required for a local Qdrant instance.

Output layout:
    data/snapshots/
        tmdb_movies_<timestamp>.snapshot
        tmdb_reviews_<timestamp>.snapshot

Run this against a populated local Qdrant before packaging for distribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx


COLLECTIONS = ["tmdb_movies", "tmdb_reviews"]


def trigger_snapshot(base_url: str, collection: str, client: httpx.Client) -> str:
    """Trigger snapshot creation; return the snapshot name."""
    resp = client.post(
        f"{base_url}/collections/{collection}/snapshots",
        timeout=120.0,
    )
    resp.raise_for_status()
    result = resp.json()
    # Qdrant returns {"result": {"name": "...", ...}, "status": "ok", "time": ...}
    name: str = result["result"]["name"]
    return name


def download_snapshot(
    base_url: str,
    collection: str,
    snapshot_name: str,
    out_path: Path,
    client: httpx.Client,
) -> None:
    """Stream-download a snapshot file."""
    url = f"{base_url}/collections/{collection}/snapshots/{snapshot_name}"
    with client.stream("GET", url, timeout=300.0) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                fh.write(chunk)


def snapshot_collection(
    base_url: str,
    collection: str,
    out_dir: Path,
    client: httpx.Client,
) -> Path:
    print(f"  [{collection}] triggering snapshot creation ...", flush=True)
    name = trigger_snapshot(base_url, collection, client)
    print(f"  [{collection}] snapshot name: {name}", flush=True)

    dest = out_dir / name
    if dest.exists():
        print(f"  [{collection}] already present at {dest}, skipping download.")
        return dest

    print(f"  [{collection}] downloading -> {dest} ...", flush=True)
    download_snapshot(base_url, collection, name, dest, client)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [{collection}] done ({size_mb:.1f} MB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot local Qdrant collections to data/snapshots/."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant base URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/snapshots",
        help="Directory to write snapshot files (default: data/snapshots)",
    )
    args = parser.parse_args()

    base_url: str = args.url.rstrip("/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify Qdrant is reachable
    with httpx.Client() as probe:
        try:
            r = probe.get(f"{base_url}/healthz", timeout=10.0)
            r.raise_for_status()
        except Exception as exc:
            print(f"ERROR: Qdrant not reachable at {base_url}: {exc}", file=sys.stderr)
            print(
                "Make sure the qdrant service is running: docker compose up -d qdrant",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Qdrant reachable at {base_url}")
    print(f"Output directory: {out_dir.resolve()}\n")

    paths: list[Path] = []
    with httpx.Client() as client:
        for collection in COLLECTIONS:
            # Confirm collection exists
            r = client.get(
                f"{base_url}/collections/{collection}", timeout=15.0
            )
            if r.status_code == 404:
                print(
                    f"  [{collection}] NOT FOUND — skipping (is ingest complete?)",
                    file=sys.stderr,
                )
                continue
            r.raise_for_status()
            count = r.json()["result"]["points_count"]
            print(f"  [{collection}] {count:,} points")

            path = snapshot_collection(base_url, collection, out_dir, client)
            paths.append(path)

    if not paths:
        print("\nERROR: no snapshots produced.", file=sys.stderr)
        sys.exit(1)

    print("\nSnapshots written:")
    for p in paths:
        print(f"  {p}")
    print(
        "\nTo restore these into a fresh local Qdrant, run:\n  make restore-seed"
    )


if __name__ == "__main__":
    main()
