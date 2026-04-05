"""
Download FineWebEdu train/val shards from Backblaze B2 to /workspace.

Usage:
  python b2_download_shards.py \
    --key-id 0053541b80ef3350000000001 \
    --app-key K005nBhlarJEEgZn6axCgXXYLqaqckM \
    --bucket runpod1231 \
    --local-root /workspace/foundry-llm-runtime/artifacts/p15/finewebedu_sample10bt/canonical \
    --workers 16

Downloads:
  shards_train/  -> <local-root>/shards_train/
  shards_val/    -> <local-root>/shards_val/       (only if missing locally)

Skips files that already exist with the correct size.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from queue import Queue
from typing import Any

# ─── B2 API helpers ───────────────────────────────────────────────────────────

def _b2_request(url: str, *, headers: dict, data: dict | None = None) -> Any:
    req = urllib.request.Request(
        url,
        headers=headers,
        data=json.dumps(data).encode() if data is not None else None,
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def b2_authorize(key_id: str, app_key: str) -> dict:
    cred = base64.b64encode(f"{key_id}:{app_key}".encode()).decode()
    return _b2_request(
        "https://api.backblazeb2.com/b2api/v2/b2_authorize_account",
        headers={"Authorization": f"Basic {cred}"},
    )


def b2_list_all_files(*, api_url: str, token: str, bucket_id: str, prefix: str = "") -> list[dict]:
    all_files: list[dict] = []
    next_name = None
    while True:
        body: dict = {"bucketId": bucket_id, "maxFileCount": 1000, "prefix": prefix}
        if next_name:
            body["startFileName"] = next_name
        result = _b2_request(
            f"{api_url}/b2api/v2/b2_list_file_names",
            headers={"Authorization": token},
            data=body,
        )
        files = result.get("files", [])
        all_files.extend(files)
        next_name = result.get("nextFileName")
        print(f"  [list] {prefix!r}: fetched {len(all_files)} so far...", flush=True)
        if not next_name or not files:
            break
    return all_files


def b2_get_download_url(*, download_base_url: str, bucket_name: str, file_name: str, token: str) -> str:
    return f"{download_base_url}/file/{bucket_name}/{file_name}"


# ─── Download worker ──────────────────────────────────────────────────────────

_RETRY_DELAYS = [2, 5, 15, 30]  # seconds between retries


def _download_one(
    *,
    url: str,
    auth_header: str,
    dest_path: Path,
    expected_size: int,
    progress_lock: threading.Lock,
    progress: dict,
) -> None:
    # Skip if already complete
    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        with progress_lock:
            progress["skipped"] += 1
            progress["bytes_skipped"] += expected_size
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            req = urllib.request.Request(url, headers={"Authorization": auth_header})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) != expected_size:
                raise ValueError(f"size mismatch: got {len(data)}, expected {expected_size}")
            tmp_path.write_bytes(data)
            tmp_path.rename(dest_path)
            with progress_lock:
                progress["done"] += 1
                progress["bytes_done"] += expected_size
            return
        except Exception as exc:
            with progress_lock:
                progress["errors"] += 1
            if attempt == len(_RETRY_DELAYS):
                with progress_lock:
                    progress["failed"].append(str(dest_path))
                print(f"  [FAIL] {dest_path.name}: {exc}", flush=True)
                return
            print(f"  [retry {attempt+1}] {dest_path.name}: {exc}", flush=True)


def _worker(
    q: Queue,
    auth_header: str,
    progress_lock: threading.Lock,
    progress: dict,
) -> None:
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        url, dest_path, expected_size = item
        _download_one(
            url=url,
            auth_header=auth_header,
            dest_path=dest_path,
            expected_size=expected_size,
            progress_lock=progress_lock,
            progress=progress,
        )
        q.task_done()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Download B2 shards to local workspace.")
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--app-key", required=True)
    parser.add_argument("--bucket", default="runpod1231")
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("/workspace/foundry-llm-runtime/artifacts/p15/finewebedu_sample10bt/canonical"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--train-only", action="store_true", help="Skip val shards (val already in image)")
    args = parser.parse_args()

    print("=" * 70)
    print("B2 SHARD DOWNLOADER")
    print(f"  bucket:     {args.bucket}")
    print(f"  local-root: {args.local_root}")
    print(f"  workers:    {args.workers}")
    print(f"  train-only: {args.train_only}")
    print("=" * 70)

    # ── Authorize ─────────────────────────────────────────────────────────────
    print("[1/4] Authorizing with B2...", flush=True)
    auth = b2_authorize(args.key_id, args.app_key)
    api_url: str = auth["apiUrl"]
    token: str = auth["authorizationToken"]
    download_url: str = auth["downloadUrl"]
    account_id: str = auth["accountId"]
    print(f"  accountId:   {account_id}")
    print(f"  apiUrl:      {api_url}")
    print(f"  downloadUrl: {download_url}")

    # ── Find bucket ID ────────────────────────────────────────────────────────
    print("[2/4] Looking up bucket ID...", flush=True)
    buckets_resp = _b2_request(
        f"{api_url}/b2api/v2/b2_list_buckets?accountId={account_id}",
        headers={"Authorization": token},
    )
    bucket_id = None
    for b in buckets_resp.get("buckets", []):
        if b["bucketName"] == args.bucket:
            bucket_id = b["bucketId"]
            break
    if not bucket_id:
        print(f"ERROR: bucket {args.bucket!r} not found", file=sys.stderr)
        return 1
    print(f"  bucket_id: {bucket_id}")

    # ── List files ────────────────────────────────────────────────────────────
    print("[3/4] Listing files...", flush=True)
    prefixes = ["shards_train/"]
    if not args.train_only:
        prefixes.append("shards_val/")

    file_map: list[tuple[str, Path, int]] = []  # (url, dest_path, expected_size)
    for prefix in prefixes:
        print(f"  Listing prefix: {prefix}", flush=True)
        files = b2_list_all_files(api_url=api_url, token=token, bucket_id=bucket_id, prefix=prefix)

        subdir_name = prefix.rstrip("/")  # "shards_train" or "shards_val"

        for f in files:
            if f.get("action") != "upload":
                continue
            b2_name: str = f["fileName"]
            size: int = f["contentLength"]
            # Map: shards_train/train/foo.bin -> <local_root>/shards_train/train/foo.bin
            relative = b2_name  # already has shards_train/ prefix
            local_path = args.local_root / relative
            url = f"{download_url}/file/{args.bucket}/{b2_name}"
            file_map.append((url, local_path, size))

        total_listed = len([x for x in file_map if subdir_name in str(x[1])])
        total_size = sum(x[2] for x in file_map if subdir_name in str(x[1]))
        print(f"  {prefix}: {total_listed} files, {total_size/1024/1024/1024:.2f} GB", flush=True)

    total_files = len(file_map)
    total_bytes = sum(x[2] for x in file_map)
    print(f"\n  Total: {total_files} files, {total_bytes/1024/1024/1024:.2f} GB")

    # Check disk space
    stat = os.statvfs("/workspace")
    free_bytes = stat.f_bavail * stat.f_frsize
    print(f"  Free on /workspace: {free_bytes/1024/1024/1024:.2f} GB")
    if free_bytes < total_bytes * 1.05:
        print(f"WARNING: only {free_bytes/1024/1024/1024:.1f} GB free, need {total_bytes/1024/1024/1024:.1f} GB")

    # ── Download ──────────────────────────────────────────────────────────────
    print("\n[4/4] Downloading...", flush=True)
    print(f"  Using {args.workers} parallel workers", flush=True)

    progress_lock = threading.Lock()
    progress: dict = {"done": 0, "skipped": 0, "failed": [], "errors": 0,
                       "bytes_done": 0, "bytes_skipped": 0}

    q: Queue = Queue(maxsize=args.workers * 4)
    threads = []
    for _ in range(args.workers):
        t = threading.Thread(
            target=_worker,
            args=(q, token, progress_lock, progress),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Progress reporter
    start_time = time.time()

    def _reporter():
        last_done = 0
        while True:
            time.sleep(30)
            with progress_lock:
                done = progress["done"]
                skipped = progress["skipped"]
                failed_count = len(progress["failed"])
                bdone = progress["bytes_done"]
                bskip = progress["bytes_skipped"]
            elapsed = time.time() - start_time
            total_done = done + skipped
            pct = 100 * total_done / max(1, total_files)
            rate = (bdone - (last_done * 1_912_680)) / 30 / 1024 / 1024  # rough MB/s
            eta = (total_files - total_done) * (elapsed / max(1, total_done)) if total_done > 0 else 0
            print(
                f"  [progress] {total_done}/{total_files} ({pct:.1f}%) | "
                f"new={done} skip={skipped} fail={failed_count} | "
                f"downloaded={bdone/1024/1024/1024:.2f}GB | "
                f"elapsed={elapsed/60:.1f}min | ETA={eta/60:.1f}min",
                flush=True,
            )
            last_done = done
            if total_done >= total_files:
                break

    reporter = threading.Thread(target=_reporter, daemon=True)
    reporter.start()

    # Enqueue all work
    for url, dest_path, size in file_map:
        q.put((url, dest_path, size))

    # Signal workers to stop
    for _ in range(args.workers):
        q.put(None)

    q.join()

    elapsed = time.time() - start_time
    with progress_lock:
        done = progress["done"]
        skipped = progress["skipped"]
        failed = progress["failed"]

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print(f"  Downloaded: {done} files")
    print(f"  Skipped (already present): {skipped} files")
    print(f"  Failed: {len(failed)} files")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    if failed:
        print("  FAILED FILES:")
        for f in failed[:20]:
            print(f"    {f}")

    # ── Verify manifest ───────────────────────────────────────────────────────
    manifest_path = args.local_root / "shards_train" / "tinyllama_p15_shards_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            n_shards = len(manifest.get("shard_inventory", []))
            print(f"\n  Manifest: {n_shards} shards declared")
        except Exception as e:
            print(f"  Manifest read error: {e}")

    bin_count = sum(1 for _, p, _ in file_map if str(p).endswith(".bin"))
    actual_bins = list((args.local_root / "shards_train" / "train").glob("*.bin"))
    print(f"  Expected .bin files: {bin_count}")
    print(f"  Actual .bin files on disk: {len(actual_bins)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
