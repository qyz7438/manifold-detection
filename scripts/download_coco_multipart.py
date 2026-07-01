"""Multi-segment COCO 2017 downloader with per-segment resume and verification.

Uses standard-library urllib so no extra dependencies are needed.
Run:
    python scripts/download_coco_multipart.py --out-dir E:/CLIproject/RLimage/data/coco --workers 12
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


FILES = [
    ("train2017.zip", "http://images.cocodataset.org/zips/train2017.zip"),
    ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
    ("annotations_trainval2017.zip", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
]


def _get_length(url: str) -> int | None:
    req = Request(url, method="HEAD")
    with urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0)) or None


def _download_segment(url: str, out_path: Path, start: int, end: int) -> None:
    """Download bytes [start, end) into a segment file with resume/retry."""
    segment_path = Path(str(out_path) + f".part_{start}_{end}")
    expected = end - start
    max_retries = 20
    for attempt in range(max_retries):
        existing = segment_path.stat().st_size if segment_path.exists() else 0
        if existing == expected:
            return
        if existing > expected:
            # Truncate accidental overflow.
            with open(segment_path, "rb+") as f:
                f.truncate(expected)
            return

        req_start = start + existing
        headers = {"Range": f"bytes={req_start}-{end - 1}"}
        req = Request(url, headers=headers)
        try:
            with open(segment_path, "ab") as f, urlopen(req, timeout=120) as resp:
                while True:
                    chunk = resp.read(2 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    existing += len(chunk)
                    if existing >= expected:
                        break
        except Exception as exc:
            print(f"  segment {start}-{end} attempt {attempt + 1} error: {exc}, retrying...")
            time.sleep(5)

    # Final size check.
    final = segment_path.stat().st_size if segment_path.exists() else 0
    if final != expected:
        raise RuntimeError(f"Segment {start}-{end} final size {final} != expected {expected}")


def _concat_segments(out_path: Path, segments: list[tuple[int, int]]) -> None:
    with open(out_path, "wb") as out_f:
        for start, end in segments:
            segment_path = Path(str(out_path) + f".part_{start}_{end}")
            expected = end - start
            actual = segment_path.stat().st_size
            if actual != expected:
                raise RuntimeError(f"Segment {segment_path} size {actual} != expected {expected}")
            with open(segment_path, "rb") as seg_f:
                while True:
                    chunk = seg_f.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
            segment_path.unlink(missing_ok=True)


def download_file(name: str, url: str, out_dir: Path, workers: int) -> None:
    out_path = out_dir / name
    if out_path.exists():
        print(f"[skip] {name} already exists")
        return

    length = _get_length(url)
    if length is None or length <= 0:
        raise RuntimeError(f"Could not determine size of {url}")
    print(f"[start] {name}: {length / (1024 ** 3):.2f} GB, {workers} segments")

    segment_size = length // workers
    segments = []
    for i in range(workers):
        start = i * segment_size
        end = length if i == workers - 1 else (i + 1) * segment_size
        segments.append((start, end))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_download_segment, url, out_path, s, e): (s, e)
            for s, e in segments
        }
        for fut in as_completed(futures):
            s, e = futures[fut]
            try:
                fut.result()
                print(f"  segment {s}-{e} done")
            except Exception as exc:
                print(f"  segment {s}-{e} failed: {exc}")
                raise

    _concat_segments(out_path, segments)
    elapsed = time.time() - t0
    print(f"[done] {name} in {elapsed / 60:.1f} min, avg {length / elapsed / 1024:.1f} KB/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="E:/CLIproject/RLimage/data/coco")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--files", default="all", help="Comma-separated names or 'all'")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_list = FILES if args.files == "all" else [f for f in FILES if f[0] in args.files.split(",")]
    for name, url in file_list:
        try:
            download_file(name, url, out_dir, args.workers)
        except Exception as exc:
            print(f"[error] {name}: {exc}")
            raise


if __name__ == "__main__":
    main()
