# explanation:

"""
dataloader.py  –  clip-based data pipeline for ObjectDetector (see model.py)
 
Each dataset sample is one Clip of CLIP_LENGTH frames.
__getitem__ returns:
    motion_vectors : [T, 4, H_tokens, W_tokens]   (float32)   – if USE_MOTIONVECTORS
    residuals      : [T, 3, H, W]                  (float32, normalised to [0,1])  – if USE_RESIDUALS
    frame_types    : list[str]  length T
    boxes          : [T, 2]    float32  [xmin, xmax] per frame
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
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import asdict

from data_classes import Frame, Clip, MV_STRUCT
from import_data import import_clip

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
    
    return Frame(
        motion_vectors      = _random_mv_grid(),
        frame_type          = frame_type or random.choice(["I", "P", "B"]),
        residuals           = np.random.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8),
        true_bounding_boxes = [xmin, xmax],
        true_class          = random.randint(0, NUM_CLASSES - 1),
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
        clips:       list[Clip],
        indices:     list[int],
        base_mv_scale: int = BASE_MV_SCALE,
    ):
        self.clips         = clips
        self.indices       = indices
        self.base_mv_scale = base_mv_scale
 
    def __len__(self) -> int:
        return len(self.indices)
 
    def __getitem__(self, idx: int) -> dict:
        clip: Clip = self.clips[self.indices[idx]]

        # Extract motion vectors, residuals, and annotations from the clip
        mv_list     = []
        res_list    = []
        frame_types = []
        boxes_list  = []
        cls_list    = []

        # Iterate over frames in the clip
        for frame in clip.frames:

            if USE_MOTIONVECTORS:
                # MV_STRUCT flat array [H_tokens * W_tokens] → [4, H_tokens, W_tokens]
                n_mv   = len(frame.motion_vectors)
                h_mv   = int(round(n_mv ** 0.5))   # assumes square token grid
                w_mv   = n_mv // h_mv
                mv_grid = np.stack([
                    frame.motion_vectors['source'].astype(np.float32),
                    frame.motion_vectors['motion_x'].astype(np.float32),
                    frame.motion_vectors['motion_y'].astype(np.float32),
                    frame.motion_vectors['motion_scale'].astype(np.float32),
                ], axis=0).reshape(4, h_mv, w_mv)
                mv_list.append(torch.from_numpy(mv_grid))

            if USE_RESIDUALS:
                # [H, W, 3] uint8 → [3, H, W] float32 in [0, 1]
                res = torch.from_numpy(
                    frame.residuals.astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                res_list.append(res)

            # annotations (always included)
            frame_types.append(frame.frame_type)
            boxes_list.append(torch.tensor(frame.true_bounding_boxes, dtype=torch.float32))
            cls_list.append(torch.tensor(frame.true_class, dtype=torch.long))

        sample = {
            "frame_types": frame_types,                      # list[str], length T
            "boxes":       torch.stack(boxes_list),          # [T, 2]
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
        "boxes":       torch.stack([s["boxes"]      for s in batch]),  # (B, T, 2)
        "true_class":  torch.stack([s["true_class"] for s in batch]),  # (B, T)
    }
    if USE_MOTIONVECTORS:
        collated["motion_vectors"] = torch.stack([s["motion_vectors"] for s in batch])
    if USE_RESIDUALS:
        collated["residuals"]      = torch.stack([s["residuals"]      for s in batch])

    return collated


# helper to create dataset splits
def _make_splits(n: int) -> dict[str, list[int]]:
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return {
        "train": indices[:n_train],
        "val":   indices[n_train : n_train + n_val],
        "test":  indices[n_train + n_val :],
    }



#build dataloader?
def build_data_loaders(
    clips: list[Clip] | None = None,
    npz_dir: str | None = None,
    clip_length: int = CLIP_LENGTH,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Three ways to call this:
 
        build_data_loaders(npz_dir="/app/test/output_dir")
            -> loads every .npz produced by extractor.py via import_data.import_clip()
               this is the normal production path, and also used for training?
 
        build_data_loaders(clips=my_clip_list)
            -> uses a list of Clip objects already in memory

        build_data_loaders()
            -> generates 200 synthetic dummy clips for testing
 
    Returns (train_loader, val_loader, test_loader).
    """
    if clips is not None:
        pass  # use as-is
    elif npz_dir is not None:
        npz_files = sorted(
            os.path.join(npz_dir, f)
            for f in os.listdir(npz_dir)
            if f.endswith(".npz")
        )
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {npz_dir}")
        print(f"[DataLoader] Loading {len(npz_files)} clips from {npz_dir} ...")
        clips = [import_clip(p) for p in npz_files]
    else:
        print("[DataLoader] No data provided - generating dummy clip dataset ...")
        clips = build_dummy_dataset(n=200, clip_length=clip_length)
 
    splits = _make_splits(len(clips))
 
    train_ds = ClipDataset(clips, splits["train"])
    val_ds   = ClipDataset(clips, splits["val"])
    test_ds  = ClipDataset(clips, splits["test"])
 
    loader_kwargs = dict(
        num_workers        = NUM_WORKERS,
        pin_memory         = NUM_WORKERS > 0,
        persistent_workers = NUM_WORKERS > 0,
        prefetch_factor    = 2 if NUM_WORKERS > 0 else None,
        collate_fn         = collate_fn,
    )
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
 
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print(f"[DataLoader] Streams active: motion_vectors={USE_MOTIONVECTORS}  residuals={USE_RESIDUALS}")

    train_loader, val_loader, test_loader = build_data_loaders()

    print(f"  Split sizes  |  train={len(train_loader.dataset)}"
          f"  val={len(val_loader.dataset)}"
          f"  test={len(test_loader.dataset)}")
    print(f"  Batches/epoch (train): {len(train_loader)}")

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
    print(f"done ({i+1} batches).\n")

    print("Iterating through all val batches …",   end=" ")
    for i, b in enumerate(val_loader):
        pass
    print(f"done ({i+1} batches).\n")

    print("Iterating through all test batches …",  end=" ")
    for i, b in enumerate(test_loader):
        pass
    print(f"done ({i+1} batches).\n")