"""
Training loop for compressed-domain object detection.
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from models import Detector


def compute_loss(predictions, gt_bboxes, gt_classes):
    """
    Compute detection loss (objectness + classification).
    predictions : (B, H, W, 5 + num_classes)
    gt_bboxes   : (B, N, 4)   ground truth [x, y, w, h], padded with zeros
    gt_classes  : (B, N)      ground truth class indices, padded with -1
    """
    pred_bbox  = predictions[..., :4]   # (B, H, W, 4) - reserved for bbox regression
    pred_conf  = predictions[..., 4]    # (B, H, W)    - objectness score per cell
    pred_class = predictions[..., 5:]   # (B, H, W, C) - class scores per cell

    # --- Objectness target ---
    # Mark the grid cell that each ground-truth bbox centre falls in as 1.0.
    target_conf = torch.zeros_like(pred_conf)
    for i in range(len(gt_classes)):
        for j in range(gt_classes.shape[1]):
            if gt_classes[i, j] < 0:   # -1 is the padding sentinel
                continue
            x, y = gt_bboxes[i, j, 0], gt_bboxes[i, j, 1]
            gx = min(int(x * pred_conf.shape[2]), pred_conf.shape[2] - 1)
            gy = min(int(y * pred_conf.shape[1]), pred_conf.shape[1] - 1)
            target_conf[i, gy, gx] = 1.0

    conf_loss = F.binary_cross_entropy_with_logits(pred_conf, target_conf)

    # --- Classification loss ---
    class_loss = torch.tensor(0.0, device=predictions.device)
    n = 0
    for i in range(len(gt_classes)):
        valid = gt_classes[i][gt_classes[i] >= 0]   # strip padding
        if len(valid) > 0:
            class_loss += F.cross_entropy(
                pred_class[i].view(-1, pred_class.shape[-1])[:len(valid)],
                valid,
            )
            n += 1
    if n > 0:
        class_loss = class_loss / n

    return conf_loss + class_loss


def train(model, train_loader, val_loader, config: Config):
    """
    Train for config.epochs epochs. Saves best checkpoint to
    config.checkpoint_dir/best.pt.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):

        # ---- training ----
        model.train()
        train_loss = 0.0
        for seqs, bboxes, classes in train_loader:
            seqs, bboxes, classes = seqs.to(device), bboxes.to(device), classes.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model(seqs), bboxes, classes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, bboxes, classes in val_loader:
                seqs, bboxes, classes = seqs.to(device), bboxes.to(device), classes.to(device)
                val_loss += compute_loss(model(seqs), bboxes, classes).item()
        val_loss /= len(val_loader)

        scheduler.step()  # step once per epoch, not per batch
        print(f"Epoch {epoch:3d}/{config.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(config.checkpoint_dir, "best.pt"))
            print("  -> saved best model")

    return model


if __name__ == "__main__":
    from dataset import get_loaders

    cfg = Config()

    # Replace with your actual list of Frame objects (from extract_vectors)
    frames = []

    train_loader, val_loader = get_loaders(
        frames,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        train_split=cfg.train_split,
        num_workers=cfg.num_workers,
    )

    model = Detector(num_classes=cfg.num_classes, hidden_channels=cfg.hidden_channels)
    train(model, train_loader, val_loader, cfg)
