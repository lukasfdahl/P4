# explanation:

"""
dataloader.py  –  clip-based data pipeline for ObjectDetector (see model.py)
 
Each dataset sample is one Clip of CLIP_LENGTH frames.
__getitem__ returns:
    motion_vectors : [T, 4, H_tokens, W_tokens]   (float32)   – if USE_MOTIONVECTORS
    residuals      : [T, 3, H, W]                  (float32, normalised to [0,1])  – if USE_RESIDUALS
    frame_types    : list[str]  length T
    boxes          : [T, 4]    float32  [xmin, xmax] per frame
    true_class     : [T]       int64    class label per frame
 
collate_fn batches those into:
    motion_vectors : [B, T, 4, H_tokens, W_tokens]  – if USE_MOTIONVECTORS
    residuals      : [B, T, 3, H, W]                – if USE_RESIDUALS
    frame_types    : list[list[str]]  shape [B][T]   (can't stack strings)
    boxes          : [B, T, 2]
    true_class     : [B, T]
"""



# libraries
import os
import glob # Added for lazy loading
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import asdict

from data_classes import Frame, Clip, MV_STRUCT
from helpers import load_clips_from_npz_dir, build_window_index

# config for experiments, to toggle input features
USE_MOTIONVECTORS = True   # include motion-vector stream
USE_RESIDUALS     = True   # include residual (RGB) stream

if not USE_MOTIONVECTORS and not USE_RESIDUALS:
    raise ValueError("At least one of USE_MOTIONVECTORS or USE_RESIDUALS must be True.")

# Config
BATCH_SIZE  = 4   # only for dummy
SEED        = 42
NUM_WORKERS = 0 #0 for test, later higher

#maybe more in training?
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# Frame / residual dimensions (H x W x C), change later?
FRAME_H, FRAME_W = 64, 64
NUM_CLASSES = 10 #dependent on dataset, 100 i think?
CLIP_LENGTH = 5 # frames per clip. matches ObjectDetector default (model.py)

# Native H.264 MV block size, determines the stored MV grid resolution.
# The multi-scale tokenizer in model.py resamples from this internally.
BASE_MV_SCALE = 16
H_TOKENS = FRAME_H // BASE_MV_SCALE   # = 4  (MV grid height)
W_TOKENS = FRAME_W // BASE_MV_SCALE   # = 4  (MV grid width)
 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# loading helper, lazy because not all in ram haha
def lazy_build_window_index(
    npz_files:      list[str],
    clip_length:    int,
    stride:         int,
    snap_to_iframe: bool = True,
    filter_empty:   bool = False,
    target_classes: list[int] | None = None,
) -> list[tuple[int, int]]:
    """
    Builds sliding windows by reading only lightweight metadata from each NPZ.

    target_classes : list[int] | None
        Original YouTube-BB class IDs (1-based) to keep.
        The downloader stores class_id - 1, so we convert here automatically.
        Windows where every frame is padding (-1) or outside the target set
        are dropped entirely.Pass None to keep all windows.
    """
    # Convert original 1-based IDs → stored 0-based IDs for O(1) lookup
    target_set = (
        {c - 1 for c in target_classes} if target_classes is not None else None
    )

    window_index = []
    filtered     = 0

    for file_idx, file_path in enumerate(npz_files):
        # mmap_mode="r" — OS only pages in the two metadata arrays we slice,
        # not the full residuals/motion_vectors arrays. Critical for big datasets
        # where loading every NPZ fully would OOM or take minutes at startup.
        data        = np.load(file_path, mmap_mode="r")
        frame_types = data["frame_types"]
        true_class  = data["true_class"]   # int32, -1 = no annotation

        n = len(frame_types)
        if n < clip_length:
            continue

        starts     = list(range(0, n - clip_length + 1, stride))
        last_valid = n - clip_length
        if starts and starts[-1] != last_valid:
            starts.append(last_valid)

        seen = set()
        for start in starts:
            if snap_to_iframe:
                search_radius = stride // 2
                best_start    = start
                for offset in range(-search_radius, search_radius + 1):
                    idx = start + offset
                    if 0 <= idx <= n - clip_length and frame_types[idx] == 'I':
                        best_start = idx
                        break
                start = best_start

            if start in seen:
                continue
            seen.add(start)

            # Drop windows with no frame belonging to a target class
            if target_set is not None:
                window_classes = set(true_class[start : start + clip_length].tolist())
                if not window_classes.intersection(target_set):
                    filtered += 1
                    continue

            window_index.append((file_idx, start))

    print(
        f"[lazy] built {len(window_index)} windows "
        f"(skipped {filtered} with no target class)"
    )
    return window_index

# Dummy-data helpers for testing rn
def _random_mv_grid(h_tokens: int = H_TOKENS, w_tokens: int = W_TOKENS) -> np.ndarray:
    """
    Returns a structured numpy array of shape [h_tokens * w_tokens] using MV_STRUCT.
    Only the 4 fields the model uses are filled (source, motion_x/y, motion_scale).
    """
    n  = h_tokens * w_tokens
    mv = np.zeros(n, dtype=MV_STRUCT)
    mv['source']       = np.random.choice([-1, 1],    n)
    mv['motion_x']     = np.random.randint(-8, 9,     n)
    mv['motion_y']     = np.random.randint(-8, 9,     n)
    mv['motion_scale'] = np.random.choice([1, 2, 4],  n)
    return mv

def _random_frame(frame_type: str | None = None) -> Frame:
    """Generates one synthetic Frame with random MV grid, residuals, and annotations to testt."""
    xmin = round(random.uniform(0.0, 0.8), 4)
    xmax = round(random.uniform(xmin, 1.0), 4)
    ymin = round(random.uniform(0.0, 0.8), 4)
    ymax = round(random.uniform(ymin, 1.0), 4)
    
    return Frame(
        motion_vectors      = _random_mv_grid(),
        frame_type          = frame_type or random.choice(["I", "P", "B"]),
        residuals           = np.random.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8),
        true_bounding_box   = (xmin, xmax, ymin, ymax),
        true_class          = random.randint(0, NUM_CLASSES - 1),
        has_object          = True
    )


def _random_clip(clip_length: int = CLIP_LENGTH) -> Clip:
    """
    Generates one synthetic Clip with a realistic frame-type pattern:
    first frame is always I (keyframe), the rest alternate P/B.
    """
    frame_types = ['I'] + ['P' if i % 2 == 0 else 'B' for i in range(clip_length - 1)]
    return Clip(frames=[_random_frame(ft) for ft in frame_types])
 
 
def build_dummy_dataset(n: int = 200, clip_length: int = CLIP_LENGTH) -> list[Clip]:
    return [_random_clip(clip_length) for _ in range(n)]


# Dataset, ClipDataset (not frames anymore)
class ClipDataset(Dataset):
    """
    Maps each Clip into the tensors ObjectDetector.forward() expects.
 
    MV layout:  structured array [H_tokens * W_tokens]
                → extract 4 channels → reshape to [4, H_tokens, W_tokens]
                → stack over T frames → [T, 4, H_tokens, W_tokens]
 
    The H_tokens / W_tokens here correspond to BASE_MV_SCALE (16).
    The multi-scale tokenizer in model.py resamples internally to finer/coarser grids.

    Which streams are active is controlled by the module-level flags
    USE_MOTIONVECTORS and USE_RESIDUALS.
    """
 
    def __init__(
        self,
        source_data: list[str] | list[Clip], # Changed to accept file paths OR dummy clips
        window_index: list[tuple[int, int]],
        indices: list[int],
        clip_length: int,
        base_mv_scale: int = BASE_MV_SCALE,
    ):
        self.source_data = source_data
        self.window_index = window_index
        self.indices = indices
        self.clip_length = clip_length
        self.base_mv_scale = base_mv_scale
        
        self.is_lazy = len(source_data) > 0 and isinstance(source_data[0], str)
        # LRU cache: keeps last 8 NPZ files open per worker (insertion-order dict)
        self._npz_cache:  dict = {}
        self._cache_size: int  = 8
 
    def __len__(self) -> int:
        return len(self.indices)
 
    def __getitem__(self, idx: int) -> dict:

        mv_list     = []
        res_list    = []
        frame_types = []
        boxes_list  = []
        cls_list    = []

        sample_idx = self.indices[idx]
        clip_idx, start = self.window_index[sample_idx]

        if self.is_lazy:
            # Direct numpy slice from NPZ — no Frame/Clip construction.
            # mmap_mode="r" means the OS only fetches the pages we slice,
            # not the whole file. LRU cache keeps last 8 files open per worker.
            file_path = self.source_data[clip_idx]
            if file_path not in self._npz_cache:
                if len(self._npz_cache) >= self._cache_size:
                    self._npz_cache.pop(next(iter(self._npz_cache)))
                self._npz_cache[file_path] = np.load(file_path, mmap_mode="r")
            npz = self._npz_cache[file_path]

            sl       = slice(start, start + self.clip_length)
            raw_mv   = npz["motion_vectors"][sl]   # [T, H, W, 1] structured
            raw_res  = npz["residuals"][sl]         # [T, H, W, 3] uint8
            raw_cls  = npz["true_class"][sl]        # [T] int32
            raw_box  = npz["boxes"][sl]             # [T, 4] float32
            raw_ft   = npz["frame_types"][sl]       # [T] str

            for t in range(self.clip_length):
                if USE_MOTIONVECTORS:
                    mv_raw = raw_mv[t]

                    # NPZ stores MVs as plain int32 — must reinterpret as structured
                    # dtype before accessing named fields (source, motion_x, etc.)
                    mv = mv_raw.view(MV_STRUCT)

                    # Squeeze trailing singleton dimension if present → [H, W]
                    if mv.ndim == 3 and mv.shape[-1] == 1:
                        mv = mv[..., 0]
                    elif mv.ndim == 1:
                        s  = int(round(len(mv) ** 0.5))
                        mv = mv.reshape(s, s)

                    # mv is now [H, W] structured → stack fields → [4, H, W]
                    mv_grid = np.stack([
                        mv['source'].astype(np.float32),
                        mv['motion_x'].astype(np.float32),
                        mv['motion_y'].astype(np.float32),
                        mv['motion_scale'].astype(np.float32),
                    ], axis=0)   # [4, H, W]

                    # Resize with pure numpy (nearest-neighbour) — avoids spawning a
                    # torch dispatch in each worker process. Only resize if needed.
                    src_h, src_w = mv_grid.shape[1], mv_grid.shape[2]
                    if src_h != H_TOKENS or src_w != W_TOKENS:
                        row_idx = (np.arange(H_TOKENS) * src_h // H_TOKENS)
                        col_idx = (np.arange(W_TOKENS) * src_w // W_TOKENS)
                        mv_grid = mv_grid[:, row_idx[:, None], col_idx[None, :]]  # [4, H_TOKENS, W_TOKENS]

                    mv_list.append(torch.from_numpy(mv_grid))

                if USE_RESIDUALS:
                    # raw_res[t] is uint8 [H, W, 3]; normalise in numpy (no copy if
                    # already float32 source), then permute to [3, H, W] via reshape.
                    frame = raw_res[t]   # [H, W, 3] uint8
                    src_h, src_w = frame.shape[0], frame.shape[1]

                    # Resize with pure numpy (bilinear approximated as nearest in worker;
                    # 64×64 frames make the quality difference negligible, and it avoids
                    # torch dispatch overhead in every worker).
                    if src_h != FRAME_H or src_w != FRAME_W:
                        row_idx = (np.arange(FRAME_H) * src_h // FRAME_H)
                        col_idx = (np.arange(FRAME_W) * src_w // FRAME_W)
                        frame = frame[row_idx[:, None], col_idx[None, :], :]  # [FRAME_H, FRAME_W, 3]

                    res = torch.from_numpy(
                        np.ascontiguousarray(frame).astype(np.float32) / 255.0
                    ).permute(2, 0, 1)   # [3, FRAME_H, FRAME_W]
                    res_list.append(res)

                frame_types.append(str(raw_ft[t]))
                boxes_list.append(torch.from_numpy(raw_box[t].astype(np.float32, copy=False)))
                cls_list.append(torch.tensor(int(raw_cls[t]), dtype=torch.long))

        else:
            # Fallback for in-memory dummy Clip objects
            clip   = self.source_data[clip_idx]
            frames = clip.frames[start:start + self.clip_length]

            for frame in frames:
                if USE_MOTIONVECTORS:
                    mv = frame.motion_vectors

                    if mv.ndim == 3 and mv.shape[-1] == 1:
                        mv = mv[..., 0]
                    elif mv.ndim == 1:
                        s  = int(round(len(mv) ** 0.5))
                        mv = mv.reshape(s, s)

                    mv_grid = np.stack([
                        mv['source'].astype(np.float32),
                        mv['motion_x'].astype(np.float32),
                        mv['motion_y'].astype(np.float32),
                        mv['motion_scale'].astype(np.float32),
                    ], axis=0)
                    mv_tensor = F.interpolate(
                        torch.from_numpy(mv_grid).unsqueeze(0),
                        size=(H_TOKENS, W_TOKENS), mode='nearest'
                    ).squeeze(0)
                    mv_list.append(mv_tensor)

                if USE_RESIDUALS:
                    res = torch.from_numpy(
                        frame.residuals.astype(np.float32) / 255.0
                    ).permute(2, 0, 1)
                    res = F.interpolate(
                        res.unsqueeze(0),
                        size=(FRAME_H, FRAME_W), mode='bilinear', align_corners=False
                    ).squeeze(0)
                    res_list.append(res)

                frame_types.append(frame.frame_type)
                boxes_list.append(torch.tensor(frame.true_bounding_box, dtype=torch.float32))
                cls_list.append(torch.tensor(frame.true_class, dtype=torch.long))

        sample = {
            "frame_types": frame_types,                      # list[str], length T
            "boxes":       torch.stack(boxes_list),          # [T, 4]
            "true_class":  torch.stack(cls_list),            # [T]
        }
        if USE_MOTIONVECTORS:
            sample["motion_vectors"] = torch.stack(mv_list)  # [T, 4, H_tokens, W_tokens]
        if USE_RESIDUALS:
            sample["residuals"]      = torch.stack(res_list) # [T, 3, H, W]

        return sample
 
# arrrange items in order
def collate_fn(batch: list[dict]) -> dict:
    #Stacks fixed-size tensors normally; boxes are now fixed (2,) so no padding needed.

    collated = {
        "frame_types": [s["frame_types"] for s in batch],
        "boxes":       torch.stack([s["boxes"]      for s in batch]),  # (B, T, 4)
        "true_class":  torch.stack([s["true_class"] for s in batch]),  # (B, T)
    }
    if USE_MOTIONVECTORS:
        collated["motion_vectors"] = torch.stack([s["motion_vectors"] for s in batch])
    if USE_RESIDUALS:
        collated["residuals"]      = torch.stack([s["residuals"]      for s in batch])

    return collated


# helper to create dataset splits
def _make_video_splits(n_videos: int) -> dict[str, set[int]]:
    video_ids = list(range(n_videos))
    random.shuffle(video_ids)

    n_train = int(n_videos * TRAIN_RATIO)
    n_val   = int(n_videos * VAL_RATIO)

    return {
        "train": set(video_ids[:n_train]),
        "val":   set(video_ids[n_train:n_train + n_val]),
        "test":  set(video_ids[n_train + n_val:]),
    }


#build dataloader?
def build_data_loaders(
    clips:          list[Clip] | None = None,
    npz_dir:        str | None = None,
    clip_length:    int = CLIP_LENGTH,
    stride:         int = CLIP_LENGTH,
    snap_to_iframe: bool = True,
    limit_1_video:  bool = False,
    max_files:      int | None = None,
    batch_size:     int = BATCH_SIZE,
    num_workers:    int = NUM_WORKERS,
    pin_memory:     bool = False,
    target_classes: list[int] | None = None,
) -> tuple[DataLoader, DataLoader | list, DataLoader | list]:
    """
    Three ways to call this:
 
        build_data_loaders(npz_dir="/app/test/output_dir")
            -> loads every .npz produced by extractor.py via lazy loading paths
               this is the normal production path, and also used for training?
 
        build_data_loaders(clips=my_clip_list)
            -> uses a list of Clip objects already in memory

        build_data_loaders()
            -> generates 200 synthetic dummy clips for testing

    target_classes : list[int] | None
        Original YouTube-BB class IDs (1-based) to keep, e.g. [1, 2, 3].
        Windows with no matching frame are skipped at index-build time.
        Class IDs are NOT remapped — class 22 stays 22, num_classes stays 23.
        Pass None to use all classes.

    Returns (train_loader, val_loader, test_loader).
    """
    if target_classes is not None:
        print(f"[DataLoader] Class filter: keeping YouTube-BB classes {sorted(target_classes)} (no remapping)")

    if clips is not None:
        source_data = clips
        if limit_1_video:
            source_data = source_data[:1]
        window_index = build_window_index(source_data, clip_length=clip_length, stride=stride, snap_to_iframe=snap_to_iframe)        
    
    elif npz_dir is not None:
        npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not npz_files:
            raise ValueError(f"No .npz files found in {npz_dir}")

        if max_files is not None:
            npz_files = npz_files[:max_files]
            
        if limit_1_video:
            npz_files = npz_files[:1]
            
        source_data  = npz_files
        window_index = lazy_build_window_index(
            source_data, clip_length=clip_length,
            stride=stride, snap_to_iframe=snap_to_iframe,
            filter_empty=True,
            target_classes=target_classes,
        )

    else:
        # Fallback dummy logic
        source_data  = build_dummy_dataset()
        window_index = build_window_index(
            source_data, clip_length=clip_length,
            stride=stride, snap_to_iframe=snap_to_iframe,
            filter_empty=True, 
        )

    n_videos = len(source_data)

    # Prevent crash and force data to train split if only 1 video
    if n_videos == 1:
        print("[DataLoader] Only 1 video detected. Forcing all clips to 'train' split to prevent crash.")
        train_idx = list(range(len(window_index)))
        val_idx   = []
        test_idx  = []
    else:
        video_splits = _make_video_splits(n_videos)
        train_idx = [i for i, (v_idx, _) in enumerate(window_index) if v_idx in video_splits["train"]]
        val_idx   = [i for i, (v_idx, _) in enumerate(window_index) if v_idx in video_splits["val"]]
        test_idx  = [i for i, (v_idx, _) in enumerate(window_index) if v_idx in video_splits["test"]]

    train_ds = ClipDataset(source_data, window_index, train_idx, clip_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    
    # Safely create val/test only if data exists
    if val_idx:
        val_ds   = ClipDataset(source_data, window_index, val_idx, clip_length)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    else:
        val_loader = []
        
    if test_idx:
        test_ds  = ClipDataset(source_data, window_index, test_idx, clip_length)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    else:
        test_loader = []

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print(f"[DataLoader] Streams active: motion_vectors={USE_MOTIONVECTORS}  residuals={USE_RESIDUALS}")

    # Set limit_1_video=True so we only grab 1 video and trigger the split fix
    train_loader, val_loader, test_loader = build_data_loaders(npz_dir="dataset/", limit_1_video=True)

    # Handle potentially empty val/test loaders cleanly in prints
    val_size = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 0
    test_size = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0

    print(f"  Split sizes  |  train={len(train_loader.dataset)}" # type: ignore
          f"  val={val_size}" 
          f"  test={test_size}") 
    print(f"  Batches (train): {len(train_loader)}")

    if len(train_loader) > 0:
        # Inspect first training batch
        batch = next(iter(train_loader))

        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key:16s} | shape {str(tuple(val.shape)):20s} | dtype {val.dtype}")
            else:
                print(f"  {key:16s} | {val}")

        print()
        print(f"  frame_types[0] (first clip): {batch['frame_types'][0]}")
        print(f"  true_class[0]  (first clip): {batch['true_class'][0].tolist()}")
        print()

        # test few times
        print("Iterating through all train batches …", end=" ")
        for i, b in enumerate(train_loader):
            pass
        print(f"done ({i+1} batches).\n") # type: ignore

        if val_loader:
            print("Iterating through all val batches …",   end=" ")
            for i, b in enumerate(val_loader):
                pass
            print(f"done ({i+1} batches).\n") # type: ignore

        if test_loader:
            print("Iterating through all test batches …",  end=" ")
            for i, b in enumerate(test_loader):
                pass
            print(f"done ({i+1} batches).\n") # type: ignore