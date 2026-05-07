"""
pack_shards.py  –  ONE-TIME step: pack existing npy_dir videos into tar shards.

Run this once on your existing dataset, then point configs at dataset/shards/.
The original npy_dirs are NOT deleted — keep them as backup or remove after verifying.

Why this fixes Ceph I/O
-----------------------
Current:  2500 videos × 5 files = 12500 small .npy files → Ceph metadata storm
After  :  ~25 tar files (~100 videos each) → Ceph streams large files fast

No new packages needed — uses only Python stdlib tarfile + numpy.

Usage
-----
  python pack_shards.py                                     # defaults: dataset/ → dataset/shards/, 100 per shard
  python pack_shards.py --n-per-shard 50                   # smaller shards (more, but less RAM per worker)
  python pack_shards.py --dry-run                          # print plan, write nothing
  python pack_shards.py --verify                           # pack then verify every shard
  python pack_shards.py --dataset /path/to/data --out /path/to/shards
"""

import argparse
import io
import os
import sys
import tarfile
import numpy as np
from pathlib import Path


REQUIRED_FILES = [
    "frame_types.npy",
    "motion_vectors.npy",
    "residuals.npy",
    "boxes.npy",
    "true_class.npy",
]


def _is_video_dir(path: str) -> bool:
    return os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, f)) for f in REQUIRED_FILES
    )


def _find_video_dirs(dataset_dir: str) -> list[str]:
    entries = sorted(os.path.join(dataset_dir, e) for e in os.listdir(dataset_dir))
    dirs    = [e for e in entries if _is_video_dir(e)]
    if not dirs:
        sys.exit(f"[pack_shards] ERROR: no npy_dirs found in {dataset_dir}")
    return dirs


def _write_npy_to_tar(tf: tarfile.TarFile, arcname: str, array: np.ndarray) -> None:
    """Serialise array to bytes in memory, then add to open tar — no temp files."""
    buf      = io.BytesIO()
    np.save(buf, array)
    raw      = buf.getvalue()
    info     = tarfile.TarInfo(name=arcname)
    info.size = len(raw)
    tf.addfile(info, io.BytesIO(raw))


def pack(dataset_dir: str, out_dir: str, n_per_shard: int, dry_run: bool = False) -> None:
    video_dirs = _find_video_dirs(dataset_dir)
    total      = len(video_dirs)
    n_shards   = (total + n_per_shard - 1) // n_per_shard
    os.makedirs(out_dir, exist_ok=True)

    print(f"[pack_shards] {total} videos → {n_shards} shards "
          f"({n_per_shard} videos/shard) → {out_dir}")

    if dry_run:
        for i in range(n_shards):
            batch = video_dirs[i * n_per_shard : (i + 1) * n_per_shard]
            print(f"  shard {i:05d}: {len(batch)} videos  "
                  f"({os.path.basename(batch[0])} … {os.path.basename(batch[-1])})")
        print("[pack_shards] DRY RUN — no files written.")
        return

    for shard_idx in range(n_shards):
        batch      = video_dirs[shard_idx * n_per_shard : (shard_idx + 1) * n_per_shard]
        shard_path = os.path.join(out_dir, f"shard-{shard_idx:05d}.tar")

        if os.path.exists(shard_path):
            print(f"  shard {shard_idx:05d} already exists — skipping")
            continue

        tmp_path = shard_path + ".tmp"
        with tarfile.open(tmp_path, "w") as tf:
            for vid_dir in batch:
                vid_id = os.path.basename(vid_dir)
                for fname in REQUIRED_FILES:
                    arr = np.load(os.path.join(vid_dir, fname), mmap_mode="r")
                    _write_npy_to_tar(tf, f"{vid_id}/{fname}", arr)

        os.rename(tmp_path, shard_path)   # atomic rename — no partial shards on crash
        size_mb = os.path.getsize(shard_path) / (1024 ** 2)
        print(f"  shard {shard_idx:05d}: {len(batch)} videos  {size_mb:.1f} MB")

    print(f"[pack_shards] Done — {n_shards} shards in {out_dir}")


def verify(dataset_dir: str, out_dir: str) -> None:
    """Read back every shard and confirm shapes/dtypes match originals."""
    shard_paths = sorted(Path(out_dir).glob("shard-*.tar"))
    if not shard_paths:
        sys.exit(f"[pack_shards] No shards found in {out_dir}")

    errors = 0
    for sp in shard_paths:
        with tarfile.open(sp, "r") as tf:
            by_vid: dict[str, list[tarfile.TarInfo]] = {}
            for m in tf.getmembers():
                vid_id = m.name.split("/")[0]
                by_vid.setdefault(vid_id, []).append(m)

            for vid_id, members in by_vid.items():
                orig_dir = os.path.join(dataset_dir, vid_id)
                for m in members:
                    fname  = m.name.split("/")[1]
                    orig   = np.load(os.path.join(orig_dir, fname), mmap_mode="r")
                    packed = np.load(io.BytesIO(tf.extractfile(m).read()))
                    if orig.shape != packed.shape or orig.dtype != packed.dtype:
                        print(f"  MISMATCH {m.name}: "
                              f"orig {orig.shape}/{orig.dtype} "
                              f"packed {packed.shape}/{packed.dtype}")
                        errors += 1
        print(f"  verified {sp.name}")

    if errors:
        print(f"[pack_shards] VERIFY FAILED: {errors} mismatches")
        sys.exit(1)
    else:
        print("[pack_shards] All shards verified OK ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",     default="dataset/",        help="npy_dir dataset root")
    ap.add_argument("--out",         default="dataset/shards/", help="Shard output directory")
    ap.add_argument("--n-per-shard", type=int, default=100,     help="Videos per shard (default 100)")
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--verify",      action="store_true",        help="Verify after packing")
    args = ap.parse_args()

    pack(args.dataset, args.out, args.n_per_shard, dry_run=args.dry_run)
    if args.verify and not args.dry_run:
        verify(args.dataset, args.out)