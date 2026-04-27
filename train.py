import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from scipy.optimize import linear_sum_assignment

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
from eval_framwork import BoundingBox, Prediction, evaluate, ModelMetric
from npz_importer   import import_clip

import mlflow
import os
from tqdm import tqdm
import pynvml


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
    pred_boxes:   torch.Tensor,   # [B, num_queries, 4]  
    pred_classes: torch.Tensor,   # [B, num_queries, num_classes]  
    gt_boxes:     torch.Tensor,   # [B, T, 4]  
    gt_classes:   torch.Tensor,   # [B, T]  
    cfg_loss:     dict,
) -> tuple[torch.Tensor, dict]:
    
    B, num_queries, num_classes = pred_classes.shape
    
    # Get weights
    w_l1   = cfg_loss.get("bbox_l1_weight",   5.0)
    w_giou = cfg_loss.get("bbox_giou_weight", 2.0)
    w_cls  = cfg_loss.get("class_weight",     1.0)

    # Probabilities for class matching cost
    probs = torch.softmax(pred_classes, dim=-1)

    total_l1   = pred_boxes.new_tensor(0.0)
    total_giou = pred_boxes.new_tensor(0.0)
    total_cls  = pred_boxes.new_tensor(0.0)
    valid_frames = 0

    for b in range(B):
        # 1. Filter out padded/empty GT frames (where class == -1)
        valid_mask = gt_classes[b] != -1
        valid_gt_boxes = gt_boxes[b][valid_mask]      # [num_gt, 4]
        valid_gt_classes = gt_classes[b][valid_mask]  # [num_gt]
        
        num_gt = valid_gt_boxes.shape[0]
        if num_gt == 0:
            continue
            
        # 2. Build the Cost Matrix [num_queries, num_gt]
        # Class cost: We want to MAXIMIZE probability, so cost is negative probability
        out_prob = probs[b] 
        cost_class = -out_prob[:, valid_gt_classes] 
        
        # L1 Bounding Box Cost: pairwise distance
        cost_l1 = torch.cdist(pred_boxes[b], valid_gt_boxes, p=1)
        
        # GIoU Cost: pairwise loop, rm loop version

        # GIoU Cost: Fully vectorized using broadcasting
        # pred_boxes[b] shape: [num_queries, 4] -> unsqueeze(1) -> [num_queries, 1, 4]
        # valid_gt_boxes shape: [num_gt, 4]     -> unsqueeze(0) -> [1, num_gt, 4]
        # giou_loss broadcasts these together to calculate the [num_queries, num_gt] matrix instantly.

        cost_giou = giou_loss(
            pred_boxes[b].unsqueeze(1),
            valid_gt_boxes.unsqueeze(0)
        )
                
        # 3. Combine into final Cost Matrix and move to CPU for SciPy
        C = (w_cls * cost_class) + (w_l1 * cost_l1) + (w_giou * cost_giou)
        C_np = C.detach().cpu().numpy()
        
        # 4. HUNGARIAN MATCHER (Bipartite Matching)
        row_ind, col_ind = linear_sum_assignment(C_np)
        
        # 5. Compute actual Loss using the optimal assignments
        for q_idx, gt_idx in zip(row_ind, col_ind):
            matched_box = pred_boxes[b, q_idx]
            matched_logit = pred_classes[b, q_idx]
            actual_gt_box = valid_gt_boxes[gt_idx]
            actual_gt_cls = valid_gt_classes[gt_idx]
            
            total_l1   += F.l1_loss(matched_box, actual_gt_box)
            total_giou += giou_loss(matched_box.unsqueeze(0), actual_gt_box.unsqueeze(0)).squeeze()
            total_cls  += F.cross_entropy(matched_logit.unsqueeze(0), actual_gt_cls.unsqueeze(0)).squeeze()
            
            valid_frames += 1

    # Safe return if the entire batch happens to be empty clips
    if valid_frames == 0:
        zero = (pred_boxes * 0).sum() + (pred_classes * 0).sum()
        return zero, {
            "loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0
        }

    # Average over valid frames
    total_l1   = total_l1   / valid_frames
    total_giou = total_giou / valid_frames
    total_cls  = total_cls  / valid_frames

    total = (w_cls * total_cls) + (w_l1 * total_l1) + (w_giou * total_giou)

    return total, {
        "loss_total": total.item(),
        "loss_cls":   (w_cls * total_cls).item(),
        "loss_l1":    (w_l1 * total_l1).item(),
        "loss_giou":  (w_giou * total_giou).item(),
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
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for batch in pbar:
        
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

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_boxes, pred_classes = model(mv, res, ft_sequence)
            loss, parts = compute_loss(pred_boxes, pred_classes, gt_boxes, gt_classes, cfg_loss)

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping, needed for stable training of transformers.  Clip to 1.0 by default, but configurable via YAML. should be tested. could also be rm if not needed...
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.get("grad_clip", 1.0))

            # optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            scaler.update()  # No backward, but still update scaler to avoid stalling if it was in a bad state


        # Update totals
        for k, v in parts.items():
            totals[k] += v
        n_batches += 1
        pbar.set_postfix(loss=f"{parts['loss_total']:.4f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


import logging

# Suppress MLflow/urllib3 noise
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

@torch.no_grad()
def validate(
    model: ObjectDetector,
    loader: torch.utils.data.DataLoader,
    cfg_loss: dict,
    device: torch.device,
    num_classes: int,
    epoch: int,
    save_dir: str = "visuals"
) -> tuple[dict, ModelMetric]:
    model.eval()
    totals = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
    n_batches = 0
    
    all_frame_preds = []
    all_frame_gts   = []
    sample_images   = []  # one image per clip (frame 0), for the overview plot

    # Temporal analysis: store full clip data from the first batch so we can
    # visualise how predictions evolve across the T frames of a single clip.
    # Each entry covers one clip and contains:
    #   "images"   : List[np.ndarray]       – all T residual frames (H,W,C)
    #   "gt_per_t" : List[List[BoundingBox]] – GT boxes at each time-step t
    #   "boxes"    : np.ndarray [Q, 4]      – clip-level predicted boxes
    #   "cls_ids"  : np.ndarray [Q]         – argmax class per query
    #   "confs"    : np.ndarray [Q]         – max softmax confidence per query
    # The model outputs one set of Q queries for the whole clip (not per frame),
    # so hopping is visible by comparing the best-matching query across each
    # frame's GT annotations over time.
    temporal_clips: list = []  # filled from batch 0 only

    for i, batch in enumerate(tqdm(loader, desc="  val  ", leave=False, unit="batch")):
        gt_boxes = batch["boxes"].to(device)
        gt_classes = batch["true_class"].to(device)
        mv = batch.get("motion_vectors")
        res = batch.get("residuals")
        ft_sequence = batch["frame_types"][0]
        
        if mv is not None: mv = mv.to(device)
        if res is not None: res = res.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_boxes, pred_logits = model(mv, res, ft_sequence)
            loss, parts = compute_loss(pred_boxes, pred_logits, gt_boxes, gt_classes, cfg_loss)

        # Accumulate losses
        for k, v in parts.items():
            totals[k] += v
        n_batches += 1

        # Prepare data for eval_framework.evaluate
        B, Q, _ = pred_boxes.shape
        probs = torch.softmax(pred_logits, dim=-1)

        # Sort box coords so xmin<=xmax and ymin<=ymax.
        # The bbox MLP+sigmoid gives values in [0,1] but with no ordering guarantee.
        # If xmin > xmax, _compute_iou() gets a negative intersection width → IoU=0
        # for every prediction regardless of how well the model has learned.
        boxes_eval = torch.stack([
            torch.minimum(pred_boxes[..., 0], pred_boxes[..., 1]),  # xmin
            torch.maximum(pred_boxes[..., 0], pred_boxes[..., 1]),  # xmax
            torch.minimum(pred_boxes[..., 2], pred_boxes[..., 3]),  # ymin
            torch.maximum(pred_boxes[..., 2], pred_boxes[..., 3]),  # ymax
        ], dim=-1)  # [B, Q, 4]

        for b in range(B):
            frame_preds = []
            frame_gts = []

            # Confidence threshold + top-K cap.
            # With num_classes=23, random softmax peaks sit around 0.04–0.15,
            # so 0.5 comfortably separates noise from genuine detections.
            # The top-K cap (default 10) prevents a flood of near-threshold
            # queries from drowning out the true positives in both the metrics
            # and the visualisation — without any spatial suppression (no NMS).
            CONF_THRESHOLD  = 0.5
            MAX_PREDS_CLIP  = 10

            confs, cls_ids = torch.max(probs[b], dim=-1)  # [Q], [Q]

            # 1. Collect all queries that clear the confidence threshold
            candidates = []
            for q in range(Q):
                conf   = confs[q].item()
                cls_id = cls_ids[q].item()
                if conf >= CONF_THRESHOLD:
                    candidates.append((conf, q, cls_id))

            # 2. Keep only the top-K by confidence
            candidates.sort(reverse=True)
            for conf, q, cls_id in candidates[:MAX_PREDS_CLIP]:
                p = Prediction(
                    xmin=boxes_eval[b, q, 0].item(),
                    xmax=boxes_eval[b, q, 1].item(),
                    ymin=boxes_eval[b, q, 2].item(),
                    ymax=boxes_eval[b, q, 3].item(),
                    class_id=cls_id,
                    confidence=conf,
                )
                frame_preds.append(p)
            
            # 2. Collect GTs for this clip.
            # Each clip has T time-steps but typically only one labelled anchor
            # frame (the rest are padding with class == -1).  Collapsing all T
            # frames into one flat list caused duplicates and made most clips
            # appear to have no GT (all -1 padding).
            # Fix: keep every distinct valid annotation, deduplicated by
            # (class_id, rounded coords) so a label repeated across frames
            # is only counted once.
            T = gt_boxes.shape[1]
            seen_gts: set = set()
            for t in range(T):
                if gt_classes[b, t] == -1:
                    continue
                cls_t  = int(gt_classes[b, t].item())
                coords = (
                    round(gt_boxes[b, t, 0].item(), 4),
                    round(gt_boxes[b, t, 1].item(), 4),
                    round(gt_boxes[b, t, 2].item(), 4),
                    round(gt_boxes[b, t, 3].item(), 4),
                )
                key = (cls_t, *coords)
                if key in seen_gts:
                    continue
                seen_gts.add(key)
                gt = BoundingBox(
                    xmin=coords[0], xmax=coords[1],
                    ymin=coords[2], ymax=coords[3],
                    class_id=cls_t,
                )
                frame_gts.append(gt)
            
            all_frame_preds.append(frame_preds)
            all_frame_gts.append(frame_gts)

            # Overview plot: store frame-0 image for first 4 clips of batch 0
            if i == 0 and b < 4 and res is not None:
                img = res[b, 0].cpu().permute(1, 2, 0).numpy()
                sample_images.append(img)

            # Temporal plot: store ALL T frames + raw query outputs for batch 0
            if i == 0 and res is not None:
                T_frames = res.shape[1]  # number of time-steps in the clip

                # All T residual frames for this clip
                clip_images = [
                    res[b, t].cpu().permute(1, 2, 0).numpy()
                    for t in range(T_frames)
                ]

                # Per-time-step GT boxes (raw, not deduplicated — we want to
                # see which frames actually have an annotation)
                gt_per_t = []
                for t in range(T_frames):
                    t_gts = []
                    if gt_classes[b, t] != -1:
                        t_gts.append(BoundingBox(
                            xmin=gt_boxes[b, t, 0].item(),
                            xmax=gt_boxes[b, t, 1].item(),
                            ymin=gt_boxes[b, t, 2].item(),
                            ymax=gt_boxes[b, t, 3].item(),
                            class_id=int(gt_classes[b, t].item()),
                        ))
                    gt_per_t.append(t_gts)

                temporal_clips.append({
                    "images":   clip_images,
                    "gt_per_t": gt_per_t,
                    "boxes":    boxes_eval[b].cpu().numpy(),   # [Q, 4]
                    "cls_ids":  cls_ids.cpu().numpy(),          # [Q]
                    "confs":    confs.cpu().numpy(),            # [Q]
                })

    # Average losses
    avg_losses = {k: v / n_batches for k, v in totals.items()}
    
    metrics = evaluate(all_frame_preds, all_frame_gts, latency=0.0)

    # Overview plot — 4 clips side-by-side (existing)
    if sample_images:
        os.makedirs(save_dir, exist_ok=True)
        vis_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
        from eval_framwork import visualize_predictions
        visualize_predictions(
            all_frame_preds[:4],
            all_frame_gts[:4],
            frame_images=sample_images,
            save_path=vis_path,
            max_preds_shown=MAX_PREDS_CLIP,
        )
        mlflow.log_artifact(vis_path, artifact_path="visuals")

    # Temporal plot — one clip unrolled across its T frames
    if temporal_clips:
        from eval_framwork import visualize_temporal

        # Each clip contains one object/class, so any scoring heuristic
        # is meaningless — just pick a random clip from the batch.
        import random as _random
        chosen = _random.randrange(len(temporal_clips))

        temp_path = os.path.join(save_dir, f"epoch_{epoch:03d}_temporal.png")
        visualize_temporal(
            clip=temporal_clips[chosen],
            conf_threshold=CONF_THRESHOLD,
            max_preds_shown=MAX_PREDS_CLIP,
            save_path=temp_path,
        )
        mlflow.log_artifact(temp_path, artifact_path="visuals")

    return avg_losses, metrics

# Main entry point
def main(config_path: str, resume: str | None = None, npz_dir_override: str | None = None) -> None:

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

    # debug to se gpu metrics different way
    pynvml.nvmlInit()
    physical_gpu_id = int(gpu_id.split(",")[0])
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_id)

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        print(f"[train] Loaded config from {config_path}")

    # Override npz_dir if passed from slurm (local scratch is much faster than /ceph)
    if npz_dir_override is not None:
        cfg["data"]["npz_dir"] = npz_dir_override
        print(f"[train] npz_dir overridden to: {npz_dir_override}")

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

    # mlflow
    exp_name = cfg.get("experiment_name", "experiment")
    
    # Optional: Set tracking URI if your MLflow server is running on a specific host/port
    # If running locally in the same folder, it defaults to a local ./mlruns directory
    mlflow.set_tracking_uri("http://localhost:8501") 
    mlflow.set_experiment(exp_name)

    # The previous `with` block closed right after log_params(), before the model
    # was built, so all log_metrics() calls in the epoch loop were outside the run.
    mlflow.start_run(run_name=cfg.get("run_name", "training_run"))
    mlflow.log_dict(cfg, "config.yaml")
    mlflow.log_params({
        "lr": cfg_train["lr"],
        "epochs": cfg_train["epochs"],
        "clip_length": cfg_data["clip_length"],
        "batch_size": cfg_train.get("batch_size", "unknown")
    })
    # Log the full config as a tag so it's readable in the MLflow run overview
    # without having to download the artifact
    mlflow.set_tag("config", yaml.dump(cfg, default_flow_style=False))
    mlflow.set_tag("gpu_id", gpu_id)

    # mixed prec:

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


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
    # pin_memory=True pages tensors into pinned (non-pageable) RAM so the
    # CUDA DMA engine can transfer them to the GPU without a CPU copy,
    # which significantly reduces the time the GPU sits idle waiting for data.
    train_loader, val_loader, test_loader = build_data_loaders(
            npz_dir=cfg["data"]["npz_dir"],
            clip_length=cfg["data"]["clip_length"],
            stride=cfg["data"]["stride"],
            snap_to_iframe=cfg["data"]["snap_to_iframe"],
            max_files=cfg["data"].get("max_files"),
            batch_size=cfg_train["batch_size"],
            num_workers=cfg_train.get("num_workers", 4),
            pin_memory=(device.type == "cuda"),
        )

    # debug
    batch = next(iter(train_loader))
    print({k: v.shape if isinstance(v, torch.Tensor) else "list"
       for k, v in batch.items()})
    
    # Safely get sizes even if val/test are empty lists
    val_size = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 0
    test_size = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0

    print(f"  Split sizes  |  train={len(train_loader.dataset)}"
          f"  val={val_size}"
          f"  test={test_size}")

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

        if val_loader:
            val_losses, val_metrics = validate(model, val_loader, cfg_loss, device, cfg_model["num_classes"], epoch)
        else:
            print("  [train] No validation data, skipping validate()")
            # use dummy values so the rest of the loop doesn't crash
            val_losses = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
            val_metrics = None

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
        )

        metrics_to_log = {
                "train_loss_total": train_losses["loss_total"],
                "train_loss_cls": train_losses["loss_cls"],
                "train_loss_l1": train_losses["loss_l1"],
                "train_loss_giou": train_losses["loss_giou"],
                "val_loss_total": val_losses["loss_total"],
                "learning_rate": lr_now
            }

        if val_loader:
            metrics_to_log.update({
                "val_accuracy": val_metrics.accuracy,
                "val_mAP_50": val_metrics.mAP_50,
                "val_IoU": val_metrics.iou
            })

        # new method to try to get the gpu metrics
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        
        metrics_to_log.update({
            "my_gpu/memory_used_MB": mem_info.used // (1024 ** 2),
            "my_gpu/memory_total_MB": mem_info.total // (1024 ** 2),
            "my_gpu/utilization_percent": util_info.gpu
        })

        mlflow.log_metrics(metrics_to_log, step=epoch)

        mlflow.log_metrics(metrics_to_log, step=epoch)
        
        if val_loader:
            print(
            f"| acc {val_metrics.accuracy:.3f} "
            f"| mAP50 {val_metrics.mAP_50:.3f} "
            f"| IoU {val_metrics.iou:.3f} "
            )

        print(
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

            mlflow.log_artifact(best_path, artifact_path="models")

    print(f"\n[train] Done. Best val_loss: {best_val_loss:.4f}")
    mlflow.end_run()
    pynvml.nvmlShutdown()


# main loop where all is actiaveted
#could be paired with checks firstly
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--npz-dir",  default=None,
                        help="Override data.npz_dir from config (e.g. local scratch path)")
    args = parser.parse_args()

    with open(args.config) as f:
        _cfg = yaml.safe_load(f)

    main(
        config_path = args.config,
        resume      = _cfg.get("resume", None),
        npz_dir_override = args.npz_dir,
    )


#mlflow server --host 127.0.0.1 --port 8080 
#for local pc tetst in other terminal

#kill 
# lsof -ti:8080 | xargs kill -9