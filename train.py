import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from model      import ObjectDetector
from dataloader import build_data_loaders
from helpers import (
    clips_from_long_videos,
    save_checkpoint,
    load_checkpoint,
    make_warmup_scheduler,
    xyxy_to_xywh,
    load_clips_from_npz_dir,
)
from eval_framwork import BoundingBox, Prediction, evaluate
from npz_importer   import import_clip



#list of things for train.py
#loss for bounding boxes: L1 + GIoU
# loss for classes: Cross-entropy
# train  epoch function
# validate
# test?
# main

# loss 
def giou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU loss for [xmin, xmax, ymin, ymax] normalised boxes.

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box of A and B.
    Loss = 1 - GIoU  (in [0, 2]).

    Shapes: [..., 4]  →  [...] scalar per box pair.
    """
    # Unpack both boxes
    p_xmin, p_xmax = pred_boxes[..., 0], pred_boxes[..., 1]
    p_ymin, p_ymax = pred_boxes[..., 2], pred_boxes[..., 3]

    g_xmin, g_xmax = gt_boxes[..., 0], gt_boxes[..., 1]
    g_ymin, g_ymax = gt_boxes[..., 2], gt_boxes[..., 3]

    # Intersection
    ix1 = torch.max(p_xmin, g_xmin)
    iy1 = torch.max(p_ymin, g_ymin)
    ix2 = torch.min(p_xmax, g_xmax)
    iy2 = torch.min(p_ymax, g_ymax)

    inter_w     = (ix2 - ix1).clamp(min=0.0)
    inter_h     = (iy2 - iy1).clamp(min=0.0)
    intersection = inter_w * inter_h

    # Areas
    area_pred = (p_xmax - p_xmin).clamp(min=0.0) * (p_ymax - p_ymin).clamp(min=0.0)
    area_gt   = (g_xmax - g_xmin).clamp(min=0.0) * (g_ymax - g_ymin).clamp(min=0.0)
    union     = area_pred + area_gt - intersection + 1e-7

    iou = intersection / union

    # Enclosing box
    enc_x1 = torch.min(p_xmin, g_xmin)
    enc_y1 = torch.min(p_ymin, g_ymin)
    enc_x2 = torch.max(p_xmax, g_xmax)
    enc_y2 = torch.max(p_ymax, g_ymax)

    enc_area = ((enc_x2 - enc_x1).clamp(min=0.0) * (enc_y2 - enc_y1).clamp(min=0.0)) + 1e-7

    giou = iou - (enc_area - union) / enc_area
    return 1.0 - giou   # loss in [0, 2]


#compute both, cross entropy is just libary
def compute_loss(
    pred_boxes:   torch.Tensor,   # [B, num_queries, 4]  sigmoid output
    pred_classes: torch.Tensor,   # [B, num_queries, num_classes]  logits
    gt_boxes:     torch.Tensor,   # [B, T, 4]  [xmin, xmax, ymin, ymax]
    gt_classes:   torch.Tensor,   # [B, T]  int64
    cfg_loss:     dict,
) -> tuple[torch.Tensor, dict]:
    
    """
    Match each GT box to the best (highest confidence) query, then compute:
        • L1 loss on the matched box coordinates
        • GIoU loss on the matched box coordinates
        • Cross-entropy loss on the matched class logit vs true class

    The model produces one set of queries for the whole clip (T frames),
    and each frame has exactly one GT box.  We assign one unique query
    per frame by picking the query with the highest predicted confidence
    for that frame's GT class.

    Parameters
    pred_boxes   : [B, num_queries, 4]
    pred_classes : [B, num_queries, num_classes]   (raw logits)
    gt_boxes     : [B, T, 4]   normalised [xmin, xmax, ymin, ymax]
    gt_classes   : [B, T]      int64 class index per frame

    Returns
    total_loss : scalar tensor
    loss_parts : dict with individual loss values (for logging)
    """

    B, num_queries, num_classes = pred_classes.shape
    T = gt_boxes.shape[1]

    # Confidence per query = softmax then take the score for the GT class
    # Shape: [B, num_queries, num_classes]
    probs = torch.softmax(pred_classes, dim=-1)  # [B, Q, C]

    total_l1   = pred_boxes.new_tensor(0.0)
    total_giou = pred_boxes.new_tensor(0.0)
    total_cls  = pred_boxes.new_tensor(0.0)

    used_queries = pred_boxes.new_zeros(B, num_queries, dtype=torch.bool)

    valid_frames = 0

    for t in range(T):


        gt_cls_t = gt_classes[:, t]          # [B]
        gt_box_t = gt_boxes[:, t]            # [B, 4]

        # Skip frames with no object (class == -1, bbox == -1 sentinel).
        valid_mask = gt_cls_t != -1
        if not valid_mask.any():
            continue

        batch_idx = torch.arange(B, device=pred_boxes.device)[valid_mask]
        gt_cls_valid = gt_cls_t[valid_mask]
        gt_box_valid = gt_box_t[valid_mask]

        # For each batch item, find the query with the highest confidence
        # for this frame's GT class (among queries not yet assigned).
        # Shape of class_scores: [B, num_queries]
        class_scores = probs[batch_idx, :, gt_cls_valid]  # [B, Q]

        # Mask out already-assigned queries (set their score to -1)
        class_scores = class_scores.masked_fill(used_queries[valid_mask], -1.0)

        best_q = class_scores.argmax(dim=1)  # [B]

        # Mark these queries as used
        used_queries[batch_idx, best_q] = True

        # Gather predictions for the matched queries
        # pred_boxes[b, best_q[b], :] for each b
        matched_boxes   = pred_boxes[batch_idx, best_q]    # [B, 4]
        matched_logits  = pred_classes[batch_idx, best_q]  # [B, num_classes]

        # Losses
        total_l1   = total_l1   + F.l1_loss(matched_boxes, gt_box_valid) # bbox loss
        total_giou = total_giou + giou_loss(matched_boxes, gt_box_valid).mean()     # bbox loss
        total_cls  = total_cls  + F.cross_entropy(matched_logits, gt_cls_valid) # class loss

        valid_frames += 1

    if valid_frames == 0:

        zero = pred_boxes.new_tensor(0.0)
        return zero, {
            "loss_total": 0.0,
            "loss_cls": 0.0,
            "loss_l1": 0.0,
            "loss_giou": 0.0,
        }

    # Average over frames
    total_l1   = total_l1   / valid_frames
    total_giou = total_giou / valid_frames
    total_cls  = total_cls  / valid_frames

    w_l1   = cfg_loss.get("bbox_l1_weight",   5.0)
    w_giou = cfg_loss.get("bbox_giou_weight", 2.0)
    w_cls  = cfg_loss.get("class_weight",     1.0)

    total = w_cls * total_cls + w_l1 * total_l1 + w_giou * total_giou

    return total, {
        "loss_total": total.item(),
        "loss_cls":   total_cls.item(),
        "loss_l1":    total_l1.item(),
        "loss_giou":  total_giou.item(),
    }


# Train / validate one epoch
def train_one_epoch(
    model:      ObjectDetector,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    cfg_loss:   dict,
    cfg_train:  dict,
    device:     torch.device,
    scaler:     torch.cuda.amp.GradScaler
) -> dict:

    # model train
    model.train()
    totals = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
    n_batches = 0

    # batch loop
    for batch in loader:
        
        # move to device
        gt_boxes   = batch["boxes"].to(device)       # [B, T, 4]
        gt_classes = batch["true_class"].to(device)  # [B, T]
        mv         = batch.get("motion_vectors")
        res        = batch.get("residuals")
        frame_types = batch["frame_types"]            # list[list[str]], shape [B][T]

        # as we want to experiment, test with one of them
        if mv  is not None: mv  = mv.to(device)
        if res is not None: res = res.to(device)

        # frame_types for the model: it expects a flat list of length T (one per frame).
        # All clips in a batch share the same frame-type sequence (same clip_length),
        # so we take the first clip's types.
        ft_sequence = frame_types[0]  # list[str] length T

        # forward calculation
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred_boxes, pred_classes = model(mv, res, ft_sequence)
            loss, parts = compute_loss(pred_boxes, pred_classes, gt_boxes, gt_classes, cfg_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping, needed for stable training of transformers.  Clip to 1.0 by default, but configurable via YAML. should be tested. could also be rm if not needed...
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.get("grad_clip", 1.0))

        # optimizer step
        scaler.step(optimizer)
        scaler.update()


        # Update totals
        for k, v in parts.items():
            totals[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# validation step, no grad, eval framework for metrics
@torch.no_grad()
def validate(
    model:     ObjectDetector,
    loader:    torch.utils.data.DataLoader,
    cfg_loss:  dict,
    device:    torch.device,
    num_classes: int,
) -> tuple[dict, "ModelMetric"]:  # type: ignore[name-defined]
   
    """
    Run validation: compute loss and detection metrics via eval_framework.py
    """


    # model evaluation
    model.eval()
    totals   = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
    n_batches = 0

    # results storage / reset
    all_predictions:  list = []
    all_ground_truth: list = []
    latency_total = 0.0
    n_frames      = 0


    # batch loop
    for batch in loader:

        # device check
        gt_boxes    = batch["boxes"].to(device)
        gt_classes  = batch["true_class"].to(device)
        mv          = batch.get("motion_vectors")
        res         = batch.get("residuals")
        frame_types = batch["frame_types"]

        if mv  is not None: mv  = mv.to(device)
        if res is not None: res = res.to(device)

        # same as train
        # frame_types for the model: it expects a flat list of length T (one per frame).
        # All clips in a batch share the same frame-type sequence (same clip_length),
        # so we take the first clip's types.
        ft_sequence = frame_types[0]
        T = gt_boxes.shape[1]

        # forward + latency measurement
        t0 = time.perf_counter()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred_boxes, pred_classes = model(mv, res, ft_sequence)
            loss, parts = compute_loss(pred_boxes, pred_classes, gt_boxes, gt_classes, cfg_loss)

        latency_total += time.perf_counter() - t0
        n_frames      += gt_boxes.shape[0] * T

        # Loss
        for k, v in parts.items():
            totals[k] += v
        n_batches += 1

        # Build eval_framework inputs
        # pred_boxes:   [B, Q, 4]   (sigmoid, [xmin, xmax, ymin, ymax])
        # pred_classes: [B, Q, C]   (logits)
        B, Q, _ = pred_boxes.shape
        probs = torch.softmax(pred_classes, dim=-1)   # [B, Q, C]

        # Collect predictions
        for b in range(B):
            for t in range(T):
                # Ground truth
                box = gt_boxes[b, t].cpu()
                cls = gt_classes[b, t].item()

                # Skip frames with no object (class == -1, bbox == -1 sentinel)
                if cls == -1:
                    continue

                gt_frame = [BoundingBox(
                    xmin=box[0].item(), xmax=box[1].item(),
                    ymin=box[2].item(), ymax=box[3].item(),
                    class_id=int(cls),
                )]

                # Pick the top-k queries by confidence as predictions for this frame
                # We use the single highest-confidence query per frame to keep eval simple.
                conf_for_frame, best_cls_per_query = probs[b].max(dim=-1)  # [Q], [Q]
                best_q = conf_for_frame.argmax().item()

                pb = pred_boxes[b, best_q].cpu()
                pred_frame = [Prediction(
                    xmin=pb[0].item(), xmax=pb[1].item(),
                    ymin=pb[2].item(), ymax=pb[3].item(),
                    class_id=int(best_cls_per_query[best_q].item()),
                    confidence=conf_for_frame[best_q].item(),
                )]

                # Store for metrics calculation after the epoch
                all_predictions.append(pred_frame)
                all_ground_truth.append(gt_frame)

    avg_latency = latency_total / max(n_frames, 1)
    metrics     = evaluate(all_predictions, all_ground_truth, latency=avg_latency)

    return {k: v / max(n_batches, 1) for k, v in totals.items()}, metrics


# Main entry point
def main(config_path: str, resume: str | None = None) -> None:

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        print(f"[train] Loaded config from {config_path}")

    cfg_model  = cfg["model"]
    cfg_data   = cfg["data"]
    cfg_train  = cfg["training"]
    cfg_loss   = cfg["loss"]
    cfg_paths  = cfg["paths"]
    exp_name   = cfg.get("experiment_name", "experiment")

    torch.manual_seed(cfg_train.get("seed", 42))

    # config print
    print(f"[train] Config: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check for gpu, else cpu. ailab is CUDA.
    print(f"[train] Device: {device}  |  Experiment: {exp_name}") # print device and experiment name for debug

        # mixed prec:

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


    #  Build model 
    frame_h = cfg_data["frame_h"]
    frame_w = cfg_data["frame_w"]
    scales  = cfg_model["scales"]

    # from model.py
    model = ObjectDetector(
        num_classes        = cfg_model["num_classes"],
        scales             = scales,
        base_mv_scale      = cfg_model.get("base_mv_scale", 16),
        clip_length        = cfg_model["clip_length"],
        expected_h_tokens  = frame_h // min(scales),
        expected_w_tokens  = frame_w // min(scales),
        hidden_dim         = cfg_model.get("hidden_dim", 256),
        num_heads          = cfg_model.get("num_heads", 8),
        num_encoder_layers = cfg_model.get("num_encoder_layers", 4),
        num_decoder_layers = cfg_model.get("num_decoder_layers", 4),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # Data loaders
    npz_dir     = cfg_data.get("npz_dir")
    clip_length = cfg_data["clip_length"]
    stride      = cfg_data.get("stride", clip_length)
    snap        = cfg_data.get("snap_to_iframe", True)

    # set up data loaders
    train_loader, val_loader, test_loader = build_data_loaders(
            npz_dir=cfg["data"]["npz_dir"],
            clip_length=cfg["data"]["clip_length"],
            stride=cfg["data"]["stride"],   
            snap_to_iframe=cfg["data"]["snap_to_iframe"]
        )

    # debug
    batch = next(iter(train_loader))
    print({k: v.shape if isinstance(v, torch.Tensor) else "list"
       for k, v in batch.items()})
    
    print(
        f"[train] Splits  train={len(train_loader.dataset)}"  
        f"  val={len(val_loader.dataset)}"           
    )

    # configuration
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg_train["lr"],
        weight_decay = cfg_train.get("weight_decay", 1e-4),
    )

    epochs        = cfg_train["epochs"]
    warmup_epochs = cfg_train.get("warmup_epochs", 3)
    sched_type    = cfg_train.get("scheduler", "cosine")

    if sched_type == "cosine":
        main_sched = CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
    else:
        main_sched = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    scheduler = make_warmup_scheduler(optimizer, warmup_epochs, main_sched)

    # Checkpoint dir & config snapshot 
    ckpt_dir = os.path.join(cfg_paths.get("checkpoint_dir", "checkpoints"), exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save a copy of the config next to the checkpoints for full reproducibility!
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # Optionally resume if model fails
    start_epoch = 0
    best_val_loss = float("inf")

    if resume is not None:
        ckpt      = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[train] Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        train_losses = train_one_epoch(
            model, train_loader, optimizer, cfg_loss, cfg_train, device, scaler
        )

        val_losses, val_metrics = validate(
            model, val_loader, cfg_loss, device, cfg_model["num_classes"]
        )

        # Step scheduler (warm-up + cosine are both stepped per epoch)
        if sched_type == "plateau":
            main_sched.step(val_losses["loss_total"])
        else:
            scheduler.step()

        # Log training progress
        elapsed = time.time() - t_start
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:03d}/{epochs} "
            f"| train_loss {train_losses['loss_total']:.4f} "
            f"(cls {train_losses['loss_cls']:.3f} "
            f"l1 {train_losses['loss_l1']:.3f} "
            f"giou {train_losses['loss_giou']:.3f}) "
            f"| val_loss {val_losses['loss_total']:.4f} "
            f"| acc {val_metrics.accuracy:.3f} "
            f"| mAP50 {val_metrics.mAP_50:.3f} "
            f"| IoU {val_metrics.iou:.3f} "
            f"| lr {lr_now:.2e} "
            f"| {elapsed:.1f}s"
        )

        # Save last checkpoint every epoch
        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(
            {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss":  val_losses["loss_total"],
                "config":    cfg,
            },
            last_path,
        )

        # Save best checkpoint when val loss improves
        if val_losses["loss_total"] < best_val_loss:
            best_val_loss = val_losses["loss_total"]
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_loss":  best_val_loss,
                    "config":    cfg,
                },
                best_path,
            )
            print(f"  ↳ New best val_loss {best_val_loss:.4f} — saved to {best_path}")

    print(f"\n[train] Done. Best val_loss: {best_val_loss:.4f}")


# main loop where all is actiaveted
#could be paired with checks firstly
if __name__ == "__main__":
    main(
        config_path = "config.yaml",
        resume      = None,
    )
