# libraries
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import asdict

from data_classes import Frame, MotionVector

# Config
BATCH_SIZE  = 4   # only for dummy
SEED        = 42
NUM_WORKERS = 0 #0 for test, later higher

#maybe more?
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# Frame / residual dimensions (H x W x C), change later?
FRAME_H, FRAME_W = 64, 64
MAX_BOXES_PER_FRAME = 5
NUM_CLASSES = 10 #dependent on dataset, 100 i think?

# Number of MV entries per frame (variable in reality – we treat it as a flat
# feature vector here; adjust if you want to keep it as a sequence)
MV_PER_FRAME = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Dummy-data helpers
def _random_motion_vector() -> MotionVector:
    return MotionVector(
        source=random.choice([-1, 1]),
        width=random.randint(4, 16),
        height=random.randint(4, 16),
        source_x=random.randint(0, FRAME_W - 1),
        source_y=random.randint(0, FRAME_H - 1),
        destination_x=random.randint(0, FRAME_W - 1),
        destination_y=random.randint(0, FRAME_H - 1),
        motion_x=random.randint(-8, 8),
        motion_y=random.randint(-8, 8),
        motion_scale=random.choice([1, 2, 4]),
    )


def _random_frame() -> Frame:
    num_boxes = random.randint(1, MAX_BOXES_PER_FRAME)

    # BBoxes in [cx, cy, w, h] normalised to [0, 1]
    bboxes = [
        [round(random.uniform(0.1, 0.9), 4) for _ in range(4)]
        for _ in range(num_boxes)
    ]
    classes = [random.randint(0, NUM_CLASSES - 1) for _ in range(num_boxes)]
    return Frame(
        motion_vectors=[_random_motion_vector() for _ in range(MV_PER_FRAME)],
        frame_type=random.choice(["I", "P", "B"]),
        residuals=np.random.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8),
        true_bounding_boxes=bboxes,
        true_classes=classes,
    )


def build_dummy_dataset(n: int = 200) -> list[Frame]:
    return [_random_frame() for _ in range(n)]


# Dataset
class VideoFrameDataset(Dataset):

    # Map frame-type string to integer label
    FRAME_TYPE_MAP = {"I": 0, "P": 1, "B": 2, "?": 3}

    def __init__(self, frames: list[Frame], indices: list[int]):
        self.frames  = frames
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        frame: Frame = self.frames[self.indices[idx]]

        #Motion vectors → (MV_PER_FRAME, 10) float tensor
        mv_fields = [
            "source", "width", "height",
            "source_x", "source_y",
            "destination_x", "destination_y",
            "motion_x", "motion_y", "motion_scale",
        ]
        mv_array = np.array(
            [[getattr(mv, f) for f in mv_fields] for mv in frame.motion_vectors],
            dtype=np.float32,
        )
        motion_vectors = torch.from_numpy(mv_array)

        #Frame type → int
        frame_type = torch.tensor(
            self.FRAME_TYPE_MAP.get(frame.frame_type, 3), dtype=torch.long
        )

        #Residuals → (C, H, W) float tensor in [0, 1], could also be rm
        residuals = torch.from_numpy(
            frame.residuals.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        #bbox gturth
        boxes = torch.tensor(frame.true_bounding_boxes, dtype=torch.float32)

        # data classes
        classes = torch.tensor(frame.true_classes, dtype=torch.long)

        return {
            "motion_vectors": motion_vectors,   # (N_mv, 10)
            "frame_type":     frame_type,        # scalar
            "residuals":      residuals,          # (3, H, W)
            "boxes":          boxes,              # (N_obj, 4)  – variable N
            "classes":        classes,            # (N_obj,)    – variable N
        }


# arrrange items in order
def collate_fn(batch: list[dict]) -> dict:
    #Stacks fixed-size tensors normally; pads variable-length ones.

    return {
        "motion_vectors": torch.stack([s["motion_vectors"] for s in batch]),
        "frame_type":     torch.stack([s["frame_type"]     for s in batch]),
        "residuals":      torch.stack([s["residuals"]      for s in batch]),
        # pad_sequence expects list of (N_i, ...) tensors
        "boxes":    pad_sequence([s["boxes"]   for s in batch], batch_first=True),
        "classes":  pad_sequence([s["classes"] for s in batch], batch_first=True),
        "box_counts": torch.tensor([s["boxes"].shape[0] for s in batch]),
    }


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
def build_data_loaders(frames: list[Frame] | None = None,) -> tuple[DataLoader, DataLoader, DataLoader]:
    #If None, a dummy dataset of 200 synthetic frames is generated

    if frames is None:
        print("[DataLoader] No data provided – generating dummy dataset …")
        frames = build_dummy_dataset(n=200)

    splits = _make_splits(len(frames))

    train_ds = VideoFrameDataset(frames, splits["train"])
    val_ds   = VideoFrameDataset(frames, splits["val"])
    test_ds  = VideoFrameDataset(frames, splits["test"])

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
    decoded = [['I','P','B','?'][v] for v in batch['frame_type'].tolist()]
    print(f"  frame_types  (decoded) : {decoded}")
    print(f"  objects per frame      : {batch['box_counts'].tolist()}")
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