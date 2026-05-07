"""
helpers.py  —  shared utilities used across training, evaluation, and data loading.

    save_checkpoint / load_checkpoint   training state persistence
    make_warmup_scheduler               linear LR warm-up wrapping any main scheduler
    xyxy_to_xywh / xywh_to_xyxy        box format conversion between pipeline formats
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from typing import Optional


# Checkpointing
def save_checkpoint(state: dict, path: str) -> None:
    """
    Persist a training checkpoint to disk.

    state should contain:
        epoch, model, optimizer, scheduler, val_loss, config
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(state, path)
    print(f"[helpers] checkpoint saved → {path}")


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler]          = None,
) -> dict:
    """Load checkpoint and restore model / optimizer / scheduler states in-place."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[helpers] checkpoint loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# Learning-rate schedule
def make_warmup_scheduler(
    optimizer:      torch.optim.Optimizer,
    warmup_epochs:  int,
    main_scheduler: _LRScheduler,
) -> LambdaLR:
    """
    Wrap any scheduler with a linear warm-up phase.

    Epochs 0 … warmup_epochs-1: LR ramps from ~0 to base_lr.
    Epochs warmup_epochs+:      main_scheduler drives LR normally.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if epoch - warmup_epochs > 0:
            main_scheduler.step()
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# Bounding-box format conversion 
# The dataloader stores boxes as [xmin, xmax, ymin, ymax] (xyxy order).
# The eval framework expects [x, y, w, h] (xywh order).

def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """[xmin, xmax, ymin, ymax] → [x, y, w, h].  Shape: [..., 4], normalised [0,1]."""
    xmin, xmax, ymin, ymax = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([xmin, ymin, xmax - xmin, ymax - ymin], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[x, y, w, h] → [xmin, xmax, ymin, ymax].  Shape: [..., 4], normalised [0,1]."""
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x, x + w, y, y + h], dim=-1)