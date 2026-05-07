"""
data_helpers.py  —  I/O, caching, and tensor-decode utilities for the data pipeline.

Used by dataloader.py; kept separate so each piece is testable in isolation.

    _tlog                   timing log (set DEBUG_TIMING=0 to silence)
    _find_video_sources     discover npy_dirs and .npz files in a directory
    _load_source            memory-map a video's arrays
    _build_or_load_meta     per-video metadata cache (.dataset_meta.pkl)
    _build_window_index     sliding-window index with per-param disk cache
    decode_mv               structured MV array → float32 tensor [T, 4, H, W]
    decode_residuals        uint8 residual array → float32 tensor [T, 3, H, W]
"""

import os
import glob
import time
import pickle
import hashlib
import numpy as np
import torch

from data_classes import MV_STRUCT

# Timing log
_T0     = time.time()
_TIMING = os.environ.get("DEBUG_TIMING", "1") != "0"

def _tlog(msg: str) -> None:
    if _TIMING:
        print(f"[t+{time.time() - _T0:7.2f}s] {msg}", flush=True)


# Video source discovery
_REQUIRED_NPY = [
    "frame_types.npy", "motion_vectors.npy", "residuals.npy",
    "boxes.npy", "true_class.npy",
]

def _is_npy_dir(path: str) -> bool:
    return os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, f)) for f in _REQUIRED_NPY
    )

def _find_video_sources(data_dir: str) -> list[str]:
    """Return sorted list of npy_dirs and .npz files found in data_dir."""
    npy_dirs  = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if _is_npy_dir(os.path.join(data_dir, name))
    )
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    return npy_dirs + npz_files

def _load_source(path: str) -> dict:
    """Memory-map a video's arrays from an npy_dir or .npz file."""
    if os.path.isdir(path):
        return {
            k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r")
            for k in ("frame_types", "motion_vectors", "residuals", "boxes", "true_class")
        }
    return np.load(path, mmap_mode="r")


# Metadata cache (.dataset_meta.pkl)
# # Stores frame_types and true_class for every video so the window index can be
# built without loading the large motion-vector and residual arrays.
META_FNAME = ".dataset_meta.pkl"

def _build_or_load_meta(sources: list[str], data_dir: str) -> dict:
    """
    Load per-video metadata from cache if valid, otherwise scan all sources.
    Only reads the small arrays (frame_types, true_class) — never MV or residuals.
    Cache is invalidated if any source file is newer than the cache file.
    """
    meta_path = os.path.join(data_dir, META_FNAME)

    if os.path.exists(meta_path):
        meta_mtime = os.path.getmtime(meta_path)
        if all(os.path.getmtime(s) <= meta_mtime for s in sources):
            _tlog(f"[meta] loading cache ({meta_path})")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            _tlog(f"[meta] loaded — {len(meta['vid_metadata'])} videos")
            return meta

    _tlog(f"[meta] building metadata cache for {len(sources)} videos...")
    vid_metadata: dict[str, dict] = {}

    for i, path in enumerate(sources):
        vid_id = os.path.basename(path)
        data   = _load_source(path)
        vid_metadata[vid_id] = {
            "path":        path,
            "frame_types": np.array(data["frame_types"]),
            "true_class":  np.array(data["true_class"]),
        }
        if (i + 1) % 100 == 0 or (i + 1) == len(sources):
            _tlog(f"[meta]   {i+1}/{len(sources)} videos scanned")

    meta = {"vid_metadata": vid_metadata, "version": 1}

    tmp = meta_path + f".{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, meta_path)
        _tlog(f"[meta] cache written — {os.path.getsize(meta_path) // 1024 // 1024} MB")
    except Exception as e:
        if os.path.exists(tmp):
            try: os.unlink(tmp)
            except OSError: pass
        print(f"[meta] WARNING: cache write failed: {e!r} — continuing without cache")

    return meta


# Window index
def _build_window_index(
    meta:           dict,
    sources:        list[str],
    clip_length:    int,
    stride:         int,
    snap_to_iframe: bool,
    target_classes: list[int] | None,
    data_dir:       str,
    cache_sig:      str,
) -> list[tuple[int, int]]:
    """
    Build a list of (source_idx, frame_start) pairs from the metadata cache.
    Windows whose clips contain no target-class frame are dropped.
    Result is cached per parameter signature and reused on subsequent runs.
    """
    cache_path = os.path.join(data_dir, f".window_index_{cache_sig}.pkl")
    meta_path  = os.path.join(data_dir, META_FNAME)

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(meta_path):
            _tlog("[index] loading cached window index")
            with open(cache_path, "rb") as f:
                index = pickle.load(f)
            _tlog(f"[index] {len(index)} windows loaded from cache")
            return index

    _tlog("[index] building window index...")
    target_set  = {c - 1 for c in target_classes} if target_classes else None
    path_to_idx = {s: i for i, s in enumerate(sources)}
    index:   list[tuple[int, int]] = []
    filtered = skipped = 0

    for vid_id in sorted(meta["vid_metadata"].keys()):
        entry      = meta["vid_metadata"][vid_id]
        source_idx = path_to_idx.get(entry["path"])
        if source_idx is None:
            continue

        frame_types = entry["frame_types"]
        true_class  = entry["true_class"]
        n           = len(frame_types)

        if n < clip_length:
            skipped += 1
            continue

        starts     = list(range(0, n - clip_length + 1, stride))
        last_valid = n - clip_length
        if starts and starts[-1] != last_valid:
            starts.append(last_valid)

        seen = set()
        for start in starts:
            if snap_to_iframe:
                radius = stride // 2
                for offset in range(-radius, radius + 1):
                    idx = start + offset
                    if 0 <= idx <= n - clip_length and str(frame_types[idx]) == "I":
                        start = idx
                        break

            if start in seen:
                continue
            seen.add(start)

            if target_set is not None:
                window_cls = set(true_class[start:start + clip_length].tolist())
                if not window_cls.intersection(target_set):
                    filtered += 1
                    continue

            index.append((source_idx, start))

    _tlog(f"[index] built {len(index)} windows "
          f"(filtered {filtered} no-target, skipped {skipped} too-short)")

    tmp = cache_path + f".{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
        _tlog(f"[index] cached → {cache_path}")
    except Exception as e:
        if os.path.exists(tmp):
            try: os.unlink(tmp)
            except OSError: pass
        print(f"[index] WARNING: cache write failed: {e!r}")

    return index


# Tensor decode
# Both functions accept a clip-length slice [T, ...] and return a float32 tensor.
# Nearest-neighbour resize is done in numpy to avoid torch dispatch overhead in
# dataloader workers.

def decode_mv(
    raw_mv:  np.ndarray,
    h_tokens: int,
    w_tokens: int,
) -> torch.Tensor:
    """
    Convert a raw motion-vector array to a float32 tensor [T, 4, H, W].

    Accepts structured arrays of shape [T, H, W], [T, H*W], or [T, H, W, 1].
    The four channels are: source, motion_x, motion_y, motion_scale.
    """
    if raw_mv.dtype != MV_STRUCT:
        mv = raw_mv.view(MV_STRUCT)
    else:
        mv = raw_mv

    if mv.ndim == 4 and mv.shape[-1] == 1:
        mv = mv[..., 0]
    elif mv.ndim == 2:
        s  = int(round(mv.shape[1] ** 0.5))
        mv = mv.reshape(mv.shape[0], s, s)

    mv_grid = np.stack([
        mv["source"].astype(np.float32,       copy=False),
        mv["motion_x"].astype(np.float32,     copy=False),
        mv["motion_y"].astype(np.float32,     copy=False),
        mv["motion_scale"].astype(np.float32, copy=False),
    ], axis=1)                                              # [T, 4, H, W]

    src_h, src_w = mv_grid.shape[2], mv_grid.shape[3]
    if src_h != h_tokens or src_w != w_tokens:
        ri = np.arange(h_tokens) * src_h // h_tokens
        ci = np.arange(w_tokens) * src_w // w_tokens
        mv_grid = mv_grid[:, :, ri[:, None], ci[None, :]]

    return torch.from_numpy(np.ascontiguousarray(mv_grid))


def decode_residuals(
    raw_res:  np.ndarray,
    frame_h:  int,
    frame_w:  int,
) -> torch.Tensor:
    """
    Convert a raw residual array to a float32 tensor [T, 3, H, W] in [0, 1].

    Input: uint8 [T, H, W, 3].
    """
    src_h, src_w = raw_res.shape[1], raw_res.shape[2]
    if src_h != frame_h or src_w != frame_w:
        ri = np.arange(frame_h) * src_h // frame_h
        ci = np.arange(frame_w) * src_w // frame_w
        raw_res = raw_res[:, ri[:, None], ci[None, :], :]
    return torch.from_numpy(
        np.ascontiguousarray(raw_res.transpose(0, 3, 1, 2))
    ).float().div_(255.0)


def make_window_cache_sig(
    clip_length:    int,
    stride:         int,
    snap_to_iframe: bool,
    target_classes: list[int] | None,
) -> str:
    """Deterministic hash of window-index parameters for cache file naming."""
    sig_str = (f"v1|cl={clip_length}|st={stride}|snap={snap_to_iframe}"
               f"|tc={sorted(target_classes or [])}")
    return hashlib.md5(sig_str.encode()).hexdigest()[:12]