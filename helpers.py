import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from typing import Optional

from data_classes import Clip, Frame
from npz_importer import import_clip

#list of functions needed:
# sliding_window function
# save_checkpoint
# load_checkpoint

# list of functions for other scripts
# Sliding-window clipping to match with model input requirements, and checkpoint saving/loading are the main ones!
def sliding_window(
    clip:           Clip,
    clip_length:    int,
    stride:         int,
    snap_to_iframe: bool = True,
) -> list[Clip]:

    # Get the list of frames and its length
    frames = clip.frames
    n      = len(frames)

    if n < clip_length:
        return []

    # Build candidate start indices
    candidate_starts = list(range(0, n - clip_length + 1, stride))

    # Always include the last valid window so we don't drop the tail
    last_valid = n - clip_length
    if candidate_starts[-1] != last_valid:
        candidate_starts.append(last_valid)

    # Generate windows, optionally snapping starts to nearest I-frame
    windows: list[Clip] = []
    for start in candidate_starts:
        if snap_to_iframe:
            start = _snap_to_nearest_iframe(frames, start, stride)

        end = start + clip_length
        if end > n:
            # Tail window: right-align so we always get exactly clip_length frames
            end   = n
            start = end - clip_length

        windows.append(Clip(frames=frames[start:end]))

    # Deduplicate windows that collapsed to the same start after snapping
    seen:   set[int]  = set()
    unique: list[Clip] = []
    for w in windows:
        key = id(w.frames[0])   # identity of the first Frame object is unique per start
        if key not in seen:
            seen.add(key)
            unique.append(w)

    return unique


def _snap_to_nearest_iframe(frames: list[Frame], start: int, search_radius: int) -> int:
    """
    Return the index of the nearest I-frame at or after `start`, within
    `search_radius` frames.  Falls back to `start` if none is found.
    """
    end = min(start + search_radius, len(frames))
    for i in range(start, end):
        if frames[i].frame_type == 'I':
            return i
    return start

# use both sliding and snap to clip longer clips.
# model has max frame length
# we need to ensure all clips are of the same length
def clips_from_long_videos(
    long_clips:     list[Clip],
    clip_length:    int,
    stride:         int,
    snap_to_iframe: bool = True,
) -> list[Clip]:

    # Apply sliding window to each long clip and collect results
    all_windows: list[Clip] = []
    # reset to count how many videos were too short and got skipped entirely
    skipped = 0
    for i, lc in enumerate(long_clips):
        windows = sliding_window(lc, clip_length, stride, snap_to_iframe)
        if not windows:
            skipped += 1
        all_windows.extend(windows)

    print(
        f"[helpers] sliding_window: {len(long_clips)} videos → "
        f"{len(all_windows)} clips  "
        f"(clip_length={clip_length}, stride={stride}, "
        f"snap_to_iframe={snap_to_iframe}, skipped={skipped} too-short videos)"
    )
    return all_windows


# Checkpoint helpers
def save_checkpoint(state: dict, path: str) -> None:
    """
    Save a training checkpoint.

    Typical `state` dict:
        {
            "epoch":       int,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "val_loss":    float,
            "config":      dict,   # full yaml config for reproducibility
        }
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(state, path)
    print(f"[helpers] Checkpoint saved → {path}")


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer]  = None,
    scheduler: Optional[_LRScheduler]           = None,
) -> dict:

    # Load a training checkpoint and restore model/optimizer/scheduler states.
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[helpers] Checkpoint loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt

# LR warm-up scheduler because it’s a common best practice for training stability
# and it’s easy to mess up the implementation
# This scheduler wraps any main scheduler and adds a linear warm-up phase at the start of training
def make_warmup_scheduler(
    optimizer:     torch.optim.Optimizer,
    warmup_epochs: int,
    main_scheduler: _LRScheduler,
) -> LambdaLR:

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            # Linear ramp: epoch 0 → factor ≈ 0, epoch warmup_epochs-1 → factor ≈ 1
            return float(current_epoch + 1) / float(warmup_epochs)
        # After warm-up: delegate to main_scheduler
        # Step the inner scheduler one tick for every epoch past warm-up
        offset = current_epoch - warmup_epochs
        # LambdaLR multiplies base_lr by our factor, but the inner scheduler
        # already adjusts base_lr — so we read its last_lr ratio instead.
        # The cleanest way: step inner scheduler when we're called, return 1.0
        # so LambdaLR doesn't interfere.
        if offset > 0:
            main_scheduler.step()
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# bbox format conversion  (used by eval bridge in train.py) to make the data_classes/dataloader format compatible with the eval format
# easier to fix in code than to change eval file.
def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert [xmin, xmax, ymin, ymax] (data_classes / dataloader format)
    to     [x, y, w, h]             (eval-framework BoundingBox format).

    Input/output shape: [..., 4]
    All values normalised to [0, 1].
    """
    xmin = boxes[..., 0]
    xmax = boxes[..., 1]
    ymin = boxes[..., 2]
    ymax = boxes[..., 3]
    return torch.stack([xmin, ymin, xmax - xmin, ymax - ymin], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Inverse of xyxy_to_xywh.
    Convert [x, y, w, h] → [xmin, xmax, ymin, ymax].

    Input/output shape: [..., 4]
    """
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x, x + w, y, y + h], dim=-1)




def load_clips_from_npz_dir(
    npz_dir: str,
    max_files: int | None = None,
) -> list[Clip]:
    """
    Load .npz videos from a directory and convert them to long Clip objects
    without windowing to prevent RAM explosion.
    """
    npz_files = sorted(
        os.path.join(npz_dir, f)
        for f in os.listdir(npz_dir)
        if f.endswith(".npz")
    )

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    if max_files is not None:
        npz_files = npz_files[:max_files]

    print(f"[helpers] Loading {len(npz_files)} videos from {npz_dir} ...")

    long_clips = [import_clip(p) for p in npz_files]
    return long_clips




#ram isssue test

def sliding_window_indices(
    clip: Clip,
    clip_length: int,
    stride: int,
    snap_to_iframe: bool = True,
) -> list[int]:
    frames = clip.frames
    n = len(frames)

    if n < clip_length:
        return []

    candidate_starts = list(range(0, n - clip_length + 1, stride))
    last_valid = n - clip_length
    if candidate_starts[-1] != last_valid:
        candidate_starts.append(last_valid)

    starts = []
    for start in candidate_starts:
        if snap_to_iframe:
            start = _snap_to_nearest_iframe(frames, start, stride)

        end = start + clip_length
        if end > n:
            end = n
            start = end - clip_length

        starts.append(start)

    # deduplicate after snapping
    unique_starts = []
    seen = set()
    for s in starts:
        if s not in seen:
            seen.add(s)
            unique_starts.append(s)

    return unique_starts



def build_window_index(
    long_clips: list[Clip],
    clip_length: int,
    stride: int,
    snap_to_iframe: bool = True,
    filter_empty: bool = False,
) -> list[tuple[int, int]]:
    index = []
    skipped = 0
    filtered = 0

    for clip_idx, clip in enumerate(long_clips):
        frames = clip.frames
        n = len(frames)

        if n < clip_length:
            skipped += 1
            continue

        starts = list(range(0, n - clip_length + 1, stride))
        last_valid = n - clip_length
        if starts[-1] != last_valid:
            starts.append(last_valid)

        dedup = []
        seen = set()

        for start in starts:
            if snap_to_iframe:
                start = _snap_to_nearest_iframe(frames, start, stride)

            end = start + clip_length
            if end > n:
                end = n
                start = end - clip_length

            if start not in seen:
                seen.add(start)
                dedup.append(start)

        for start in dedup:
            
            if filter_empty:
                window_frames = frames[start : start + clip_length]
                if all(f.true_class == -1 for f in window_frames):
                    filtered += 1
                    continue
            
            index.append((clip_idx, start))

    print(
        f"[helpers] sliding_window: {len(long_clips)} videos → "
        f"{len(index)} clips  "
        f"(clip_length={clip_length}, stride={stride}, "
        f"snap_to_iframe={snap_to_iframe}, skipped={skipped} too-short videos), "
        f"filtered={filtered} empty)"
    )
    return index