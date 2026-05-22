"""
dataloader.py  —  PyTorch datasets and DataLoader factory for the object detector.

    ClipDataset         reads clips from npy_dirs or .npz files (production)
    DummyClipDataset    synthetic in-memory clips for smoke tests
    collate_fn          batches samples into model-ready tensors
    build_data_loaders  single entry point — auto-detects format, builds splits

Each sample dict:
    motion_vectors : [T, 4, H_tokens, W_tokens]  float32   (if use_motionvectors)
    residuals      : [T, 3, H, W]                float32   (if use_residuals)
    frame_types    : list[str]  length T
    iframe_mask    : [T]        bool
    boxes          : [T, 4]     float32
    true_class     : [T]        int64

After collation each key gains a leading batch dimension B.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from data_classes import Frame, Clip, MV_STRUCT
from data_helpers import (
    _tlog,
    _find_video_sources,
    _load_source,
    _build_or_load_meta,
    _build_window_index,
    decode_mv,
    decode_residuals,
    make_window_cache_sig,
)

# Defaults (overridden by build_data_loaders kwargs)
BATCH_SIZE    = 4
SEED          = 42
NUM_WORKERS   = 0
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1
TEST_RATIO    = 0.1
FRAME_H       = 64
FRAME_W       = 64
NUM_CLASSES   = 10
CLIP_LENGTH   = 5
BASE_MV_SCALE = 16
H_TOKENS      = FRAME_H // BASE_MV_SCALE
W_TOKENS      = FRAME_W // BASE_MV_SCALE

USE_MOTIONVECTORS = True
USE_RESIDUALS     = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Datasets
class ClipDataset(Dataset):
    """
    Reads clips from npy_dirs or .npz files via memory-mapped file handles.
    An LRU cache avoids re-opening the same file repeatedly within a worker.
    """

    def __init__(
        self,
        sources:           list[str],
        window_index:      list[tuple[int, int]],
        indices:           list[int],
        clip_length:       int,
        frame_h:           int  = FRAME_H,
        frame_w:           int  = FRAME_W,
        h_tokens:          int  = H_TOKENS,
        w_tokens:          int  = W_TOKENS,
        use_motionvectors: bool = USE_MOTIONVECTORS,
        use_residuals:     bool = USE_RESIDUALS,
        cache_size:        int  = 64,
    ):
        self.sources           = sources
        self.window_index      = window_index
        self.indices           = indices
        self.clip_length       = clip_length
        self.frame_h           = frame_h
        self.frame_w           = frame_w
        self.h_tokens          = h_tokens
        self.w_tokens          = w_tokens
        self.use_motionvectors = use_motionvectors
        self.use_residuals     = use_residuals
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return len(self.indices)

    def _get_source(self, path: str) -> dict:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        data = _load_source(path)
        self._cache[path] = data
        return data

    def __getitem__(self, idx: int) -> dict:
        source_idx, start = self.window_index[self.indices[idx]]
        data = self._get_source(self.sources[source_idx])
        sl   = slice(start, start + self.clip_length)

        frame_types = [str(ft) for ft in data["frame_types"][sl].tolist()]
        sample = {
            "frame_types": frame_types,
            "iframe_mask": torch.tensor([ft == "I" for ft in frame_types], dtype=torch.bool),
            "boxes":       torch.from_numpy(np.array(data["boxes"][sl],      dtype=np.float32, copy=True)),
            "true_class":  torch.from_numpy(np.array(data["true_class"][sl], dtype=np.int64,   copy=True)),
        }
        if self.use_motionvectors:
            sample["motion_vectors"] = decode_mv(data["motion_vectors"][sl], self.h_tokens, self.w_tokens)
        if self.use_residuals:
            sample["residuals"] = decode_residuals(data["residuals"][sl], self.frame_h, self.frame_w)
        return sample


class DummyClipDataset(Dataset):
    """Synthetic in-memory clips — for smoke tests only.  Uses the same decode
    path as ClipDataset so both paths stay in sync automatically."""

    def __init__(
        self,
        clips:             list[Clip],
        window_index:      list[tuple[int, int]],
        indices:           list[int],
        clip_length:       int,
        frame_h:           int  = FRAME_H,
        frame_w:           int  = FRAME_W,
        h_tokens:          int  = H_TOKENS,
        w_tokens:          int  = W_TOKENS,
        use_motionvectors: bool = USE_MOTIONVECTORS,
        use_residuals:     bool = USE_RESIDUALS,
    ):
        self.clips             = clips
        self.window_index      = window_index
        self.indices           = indices
        self.clip_length       = clip_length
        self.frame_h           = frame_h
        self.frame_w           = frame_w
        self.h_tokens          = h_tokens
        self.w_tokens          = w_tokens
        self.use_motionvectors = use_motionvectors
        self.use_residuals     = use_residuals

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        clip_idx, start = self.window_index[self.indices[idx]]
        frames = self.clips[clip_idx].frames[start:start + self.clip_length]

        frame_types = [f.frame_type for f in frames]
        sample = {
            "frame_types": frame_types,
            "iframe_mask": torch.tensor([ft == "I" for ft in frame_types], dtype=torch.bool),
            "boxes":       torch.tensor([list(f.true_bounding_box) for f in frames], dtype=torch.float32),
            "true_class":  torch.tensor([f.true_class               for f in frames], dtype=torch.long),
        }

        if self.use_motionvectors:
            # Stack per-frame MV arrays into [T, H, W] then use the shared decode path.
            mv_raw = np.stack([f.motion_vectors for f in frames])
            if mv_raw.ndim == 2:                     # [T, H*W] structured
                s = int(round(mv_raw.shape[1] ** 0.5))
                mv_raw = mv_raw.reshape(len(frames), s, s)
            sample["motion_vectors"] = decode_mv(mv_raw, self.h_tokens, self.w_tokens)

        if self.use_residuals:
            res_raw = np.stack([f.residuals for f in frames])   # [T, H, W, 3] uint8
            sample["residuals"] = decode_residuals(res_raw, self.frame_h, self.frame_w)

        return sample


# Collate
def collate_fn(batch: list[dict]) -> dict:
    collated = {
        "frame_types": [s["frame_types"] for s in batch],
        "iframe_mask": torch.stack([s["iframe_mask"] for s in batch]),
        "boxes":       torch.stack([s["boxes"]       for s in batch]),
        "true_class":  torch.stack([s["true_class"]  for s in batch]),
    }
    if "motion_vectors" in batch[0]:
        collated["motion_vectors"] = torch.stack([s["motion_vectors"] for s in batch])
    if "residuals" in batch[0]:
        collated["residuals"] = torch.stack([s["residuals"] for s in batch])
    return collated


# Dummy data factory
def _make_dummy_clips(n: int = 200, clip_length: int = CLIP_LENGTH) -> list[Clip]:
    def _mv_grid():
        nblocks = H_TOKENS * W_TOKENS
        mv = np.zeros(nblocks, dtype=MV_STRUCT)
        mv["source"]       = np.random.choice([-1, 1],    nblocks)
        mv["motion_x"]     = np.random.randint(-8, 9,     nblocks)
        mv["motion_y"]     = np.random.randint(-8, 9,     nblocks)
        mv["motion_scale"] = np.random.choice([1, 2, 4],  nblocks)
        return mv

    clips = []
    types = ["I"] + ["P" if i % 2 == 0 else "B" for i in range(clip_length - 1)]
    for _ in range(n):
        frames = []
        for ft in types:
            xmin = round(random.uniform(0.0, 0.8), 4)
            xmax = round(random.uniform(xmin, 1.0), 4)
            ymin = round(random.uniform(0.0, 0.8), 4)
            ymax = round(random.uniform(ymin, 1.0), 4)
            frames.append(Frame(
                motion_vectors    = _mv_grid(),
                frame_type        = ft,
                residuals         = np.random.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8),
                true_bounding_box = (xmin, xmax, ymin, ymax),
                true_class        = random.randint(0, NUM_CLASSES - 1),
                has_object        = True,
            ))
        clips.append(Clip(frames=frames))
    return clips


def _dummy_window_index(clips: list[Clip], clip_length: int, stride: int) -> list[tuple[int, int]]:
    return [
        (i, s)
        for i, clip in enumerate(clips)
        for s in range(0, len(clip.frames) - clip_length + 1, stride)
    ]


# Main entry point
# this function builds train / val / test DataLoaders
def build_data_loaders(
    npz_dir:             str   | None = None,
    clip_length:         int          = CLIP_LENGTH,
    stride:              int          = CLIP_LENGTH,
    snap_to_iframe:      bool         = True,
    max_files:           int   | None = None,
    max_files_per_class: int   | None = None,
    batch_size:          int          = BATCH_SIZE,
    num_workers:         int          = NUM_WORKERS,
    pin_memory:          bool         = False,
    target_classes:      list[int] | None = None,
    use_motionvectors:   bool         = USE_MOTIONVECTORS,
    use_residuals:       bool         = USE_RESIDUALS,
    train_ratio:         float        = TRAIN_RATIO,
    val_ratio:           float        = VAL_RATIO,
    test_ratio:          float        = TEST_RATIO,
    prefetch_factor:     int          = 2,
    persistent_workers:  bool | None  = None,
    sequential_io:       bool         = False,
) -> tuple[DataLoader, DataLoader | list, DataLoader | list]:
    """
    Build train / val / test DataLoaders.

    Pass npz_dir to use real data; omit it for synthetic dummy data (smoke tests).
    target_classes uses 1-based YouTube-BB IDs — they are NOT remapped in outputs.
    Split is done at video level to prevent data leakage between splits.

    sequential_io=True sorts train windows by source_idx so all clips from one
    folder are read together — reduces random HDD seeks at cost of shuffle.
    """

    #debug if features are not used
    if not use_motionvectors and not use_residuals:
        raise ValueError("At least one of use_motionvectors or use_residuals must be True.")

    # Dummy path
    if npz_dir is None:
        _tlog("[DataLoader] no npz_dir — using synthetic dummy data")
        clips        = _make_dummy_clips(clip_length=clip_length)
        window_index = _dummy_window_index(clips, clip_length, stride)
        n            = len(window_index)
        n_train      = int(n * train_ratio)
        n_val        = int(n * val_ratio)

        def _dl(idxs, shuffle):
            ds = DummyClipDataset(clips, window_index, idxs, clip_length,
                                  use_motionvectors=use_motionvectors,
                                  use_residuals=use_residuals)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=0, collate_fn=collate_fn)

        return (
            _dl(list(range(n_train)),            shuffle=True),
            _dl(list(range(n_train, n_train + n_val)), shuffle=False) if n_val  else [],
            _dl(list(range(n_train + n_val, n)), shuffle=False)       if n > n_train + n_val else [],
        )

    # Production path
    sources = _find_video_sources(npz_dir)
    if not sources:
        raise ValueError(f"No video sources found in {npz_dir}.")

    _tlog(f"[DataLoader] found {len(sources)} video sources in {npz_dir}")
    if target_classes:
        _tlog(f"[DataLoader] class filter: {sorted(target_classes)} (1-based, no remap)")

    if max_files is not None:
        sources = sources[:max_files]

    meta = _build_or_load_meta(sources, npz_dir)

    # Per-class video cap
    if max_files_per_class is not None and target_classes is not None:
        target_set_0 = {c - 1 for c in target_classes}
        counts: dict[int, int] = {c: 0 for c in target_set_0}
        kept = []
        for path in sources:
            vid_id  = os.path.basename(path)
            entry   = meta["vid_metadata"].get(vid_id)
            if entry is None:
                continue
            cls_arr = entry["true_class"]
            vid_cls = set(cls_arr[cls_arr != -1].tolist()) & target_set_0
            if any(counts[c] < max_files_per_class for c in vid_cls):
                kept.append(path)
                for c in vid_cls:
                    counts[c] += 1
            if all(counts[c] >= max_files_per_class for c in target_set_0):
                break
        sources = kept
        _tlog(f"[DataLoader] per-class cap {max_files_per_class}: "
              f"{ {c+1: n for c, n in counts.items()} } ({len(sources)} videos)")

    # Window index
    sig          = make_window_cache_sig(clip_length, stride, snap_to_iframe, target_classes)
    window_index = _build_window_index(
        meta, sources, clip_length, stride, snap_to_iframe, target_classes, npz_dir, sig,
    )
    if not window_index:
        raise ValueError("Window index is empty — check target_classes and dataset content.")

    # Video-level train/val/test split (fixed seed → same split every run)
    n_videos  = len(sources)
    rng       = random.Random(42)
    vid_order = list(range(n_videos))
    rng.shuffle(vid_order)
    total   = train_ratio + val_ratio + test_ratio
    n_train = int(n_videos * (train_ratio / total))
    n_val   = int(n_videos * (val_ratio   / total))
    train_vids = set(vid_order[:n_train])
    val_vids   = set(vid_order[n_train:n_train + n_val])
    test_vids  = set(vid_order[n_train + n_val:])

    train_idx = [i for i, (s, _) in enumerate(window_index) if s in train_vids]
    val_idx   = [i for i, (s, _) in enumerate(window_index) if s in val_vids]
    test_idx  = [i for i, (s, _) in enumerate(window_index) if s in test_vids]
    _tlog(f"[DataLoader] split  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    if sequential_io:
        train_idx.sort(key=lambda i: (window_index[i][0], window_index[i][1]))
        _tlog("[DataLoader] sequential_io=True — train index sorted by source folder")

    def _ds(idxs):
        return ClipDataset(
            sources, window_index, idxs, clip_length,
            use_motionvectors=use_motionvectors,
            use_residuals=use_residuals,
        )

    def _loader(idxs, shuffle: bool) -> DataLoader:
        kw: dict = {
            "batch_size":  batch_size,
            "shuffle":     shuffle,
            "num_workers": num_workers,
            "collate_fn":  collate_fn,
            "pin_memory":  pin_memory,
        }
        if num_workers > 0:
            kw["persistent_workers"] = True if persistent_workers is None else persistent_workers
            kw["prefetch_factor"]    = prefetch_factor
        return DataLoader(_ds(idxs), **kw)

    _tlog("[DataLoader] constructing DataLoaders...")
    train_loader = _loader(train_idx, shuffle=(not sequential_io))
    val_loader   = _loader(val_idx,   shuffle=False) if val_idx  else []
    test_loader  = _loader(test_idx,  shuffle=False) if test_idx else []
    _tlog("[DataLoader] DataLoaders ready")
    return train_loader, val_loader, test_loader


# test
if __name__ == "__main__":
    tr, va, _ = build_data_loaders(npz_dir="dataset/")
    print(f"train={len(tr.dataset)}  val={len(va.dataset) if hasattr(va, 'dataset') else 0}")  # type: ignore
    batch = next(iter(tr))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:16s} {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:16s} {v[0]}")