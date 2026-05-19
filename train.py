"""
train.py  —  training loop for the compressed-domain object detector.

    train_one_epoch   one forward + backward pass over the training set
    validate          eval pass with loss, mAP, and visualisation
    main              entry point: reads config, builds model/data/optimiser, runs loop
"""

import os
import time
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR

from model        import ObjectDetector
from dataloader   import build_data_loaders
from train_helpers import compute_loss, giou_loss, canonical_xyxy
from helpers      import save_checkpoint, load_checkpoint, make_warmup_scheduler, xyxy_to_xywh
from eval_framwork import BoundingBox, Prediction, evaluate, ModelMetric, visualize_temporal, visualize_predictions
import mlflow
import pynvml
from tqdm import tqdm
import logging
import argparse
import random as _random
import gc

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
        gt_boxes   = batch["boxes"].to(device, non_blocking=True)       # [B, T, 4]
        gt_classes = batch["true_class"].to(device, non_blocking=True)  # [B, T]
        mv         = batch.get("motion_vectors")
        res        = batch.get("residuals")
        iframe_mask = batch["iframe_mask"].to(device, non_blocking=True)  # [B, T]

        # as we want to experiment, test with one of them
        if mv  is not None: mv  = mv.to(device, non_blocking=True)
        if res is not None: res = res.to(device, non_blocking=True)

        # forward calculation
        # set_to_none=True drops gradient references instead of zeroing buffers —
        # saves a full GPU memset write-pass over all parameters every step.
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_boxes, pred_classes = model(mv, res, iframe_mask)
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

    # Temporal analysis: store raw GPU tensors during the loop and only call
    # .cpu() / .numpy() once after the loop ends, in a single bulk transfer.
    # This avoids per-frame CUDA sync points inside the hot validation path.
    # Stored as GPU tensors keyed per clip; converted to numpy after the loop.
    _vis_res_gpu:        list[torch.Tensor] = []  # [T, 3, H, W] per clip, batch 0
    _vis_gt_boxes_gpu:   list[torch.Tensor] = []  # [T, 4] per clip, batch 0
    _vis_gt_classes_gpu: list[torch.Tensor] = []  # [T]    per clip, batch 0
    _vis_boxes_gpu:      list[torch.Tensor] = []  # [Q, 4] per clip, batch 0
    _vis_cls_ids_gpu:    list[torch.Tensor] = []  # [Q]    per clip, batch 0
    _vis_confs_gpu:      list[torch.Tensor] = []  # [Q]    per clip, batch 0

    CONF_THRESHOLD = 0.1
    MAX_PREDS_CLIP = 10

    temporal_clips: list = []  # filled from batch 0 GPU tensors after the loop

    for i, batch in enumerate(tqdm(loader, desc="  val  ", leave=False, unit="batch")):
        gt_boxes = batch["boxes"].to(device, non_blocking=True)
        gt_classes = batch["true_class"].to(device, non_blocking=True)
        mv = batch.get("motion_vectors")
        res = batch.get("residuals")
        iframe_mask = batch["iframe_mask"].to(device, non_blocking=True)
        
        if mv is not None: mv = mv.to(device, non_blocking=True)
        if res is not None: res = res.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_boxes, pred_logits = model(mv, res, iframe_mask)
            loss, parts = compute_loss(pred_boxes, pred_logits, gt_boxes, gt_classes, cfg_loss)

        # Accumulate losses
        for k, v in parts.items():
            totals[k] += v
        n_batches += 1

        # pred_boxes/logits are now [B, T, Q, 4] / [B, T, Q, C+1]
        B, T_frames, Q, _ = pred_boxes.shape
        probs = torch.softmax(pred_logits, dim=-1)[..., :num_classes]  # [B, T, Q, C]

        boxes_eval = torch.stack([
            torch.minimum(pred_boxes[..., 0], pred_boxes[..., 1]),
            torch.maximum(pred_boxes[..., 0], pred_boxes[..., 1]),
            torch.minimum(pred_boxes[..., 2], pred_boxes[..., 3]),
            torch.maximum(pred_boxes[..., 2], pred_boxes[..., 3]),
        ], dim=-1)  # [B, T, Q, 4]

        confs_batch, cls_ids_batch = torch.max(probs, dim=-1)  # [B, T, Q]

        # Bulk CPU transfer ONCE per batch. Move everything
        # once here, then iterate over numpy arrays Python-side (free).
        confs_cpu     = confs_batch.cpu().numpy()      # [B, T, Q]
        cls_ids_cpu   = cls_ids_batch.cpu().numpy()    # [B, T, Q]
        boxes_eval_cpu = boxes_eval.cpu().numpy()      # [B, T, Q, 4]
        gt_classes_cpu = gt_classes.cpu().numpy()      # [B, T]
        gt_boxes_cpu   = gt_boxes.cpu().numpy()        # [B, T, 4]

        for b in range(B):
            for t in range(T_frames):
                confs   = confs_cpu[b, t]              # [Q] numpy
                cls_ids = cls_ids_cpu[b, t]            # [Q] numpy
                box_t   = boxes_eval_cpu[b, t]         # [Q, 4] numpy

                mask    = confs >= CONF_THRESHOLD
                cand_confs = confs[mask]
                cand_cls   = cls_ids[mask]
                cand_boxes = box_t[mask]
                topk = min(MAX_PREDS_CLIP, cand_confs.shape[0])

                frame_preds = []
                if topk > 0:
                    topk_idx = (-cand_confs).argsort()[:topk]
                    for k_idx in topk_idx:
                        frame_preds.append(Prediction(
                            xmin=float(cand_boxes[k_idx, 0]),
                            xmax=float(cand_boxes[k_idx, 1]),
                            ymin=float(cand_boxes[k_idx, 2]),
                            ymax=float(cand_boxes[k_idx, 3]),
                            class_id=int(cand_cls[k_idx]),
                            confidence=float(cand_confs[k_idx]),
                        ))

                # Per-frame GT (single box per frame)
                frame_gts = []
                gt_cls_t = int(gt_classes_cpu[b, t])
                if gt_cls_t != -1:
                    gt_box_t = gt_boxes_cpu[b, t]
                    frame_gts.append(BoundingBox(
                        xmin=float(gt_box_t[0]), xmax=float(gt_box_t[1]),
                        ymin=float(gt_box_t[2]), ymax=float(gt_box_t[3]),
                        class_id=gt_cls_t,
                    ))

                all_frame_preds.append(frame_preds)
                all_frame_gts.append(frame_gts)

            # Overview: frame-0 image from first 4 clips in batch 0
            if i == 0 and b < 4 and res is not None:
                sample_images.append(res[b, 0])

            # Temporal: stash GPU tensors for batch 0 — one clip, all T frames
            if i == 0 and res is not None:
                _vis_res_gpu.append(res[b])              # [T, 3, H, W]
                _vis_gt_boxes_gpu.append(gt_boxes[b])    # [T, 4]
                _vis_gt_classes_gpu.append(gt_classes[b])
                _vis_boxes_gpu.append(boxes_eval[b])     # [T, Q, 4]  ← now per-frame
                _vis_cls_ids_gpu.append(cls_ids_batch[b])# [T, Q]
                _vis_confs_gpu.append(confs_batch[b])    # [T, Q]


    # Single bulk CPU transfer for all visualisation tensors
    # Everything above stayed on GPU during the loop; pay the cost once.
    if sample_images:
        # sample_images is a list of [3,H,W] GPU tensors → stack → one transfer
        sample_images_np = torch.stack(sample_images).cpu().permute(0, 2, 3, 1).numpy()
        sample_images = [sample_images_np[k] for k in range(sample_images_np.shape[0])]

    if _vis_res_gpu:
        res_np      = torch.stack(_vis_res_gpu).cpu()        # [N, T, 3, H, W]
        gt_boxes_np = torch.stack(_vis_gt_boxes_gpu).cpu()   # [N, T, 4]
        gt_cls_np   = torch.stack(_vis_gt_classes_gpu).cpu() # [N, T]
        boxes_np    = torch.stack(_vis_boxes_gpu).cpu()      # [N, T, Q, 4]  ← per-frame
        cls_ids_np  = torch.stack(_vis_cls_ids_gpu).cpu()    # [N, T, Q]
        confs_np    = torch.stack(_vis_confs_gpu).cpu()      # [N, T, Q]

        for b in range(res_np.shape[0]):
            T_vis = res_np.shape[1]
            clip_images = [res_np[b, t].permute(1, 2, 0).numpy() for t in range(T_vis)]
            gt_per_t = []
            for t in range(T_vis):
                t_gts = []
                if gt_cls_np[b, t].item() != -1:
                    t_gts.append(BoundingBox(
                        xmin=float(gt_boxes_np[b, t, 0]),
                        xmax=float(gt_boxes_np[b, t, 1]),
                        ymin=float(gt_boxes_np[b, t, 2]),
                        ymax=float(gt_boxes_np[b, t, 3]),
                        class_id=int(gt_cls_np[b, t]),
                    ))
                gt_per_t.append(t_gts)
            temporal_clips.append({
                "images":   clip_images,
                "gt_per_t": gt_per_t,
                "boxes":    boxes_np[b].numpy(),    # [T, Q, 4] — per-frame
                "cls_ids":  cls_ids_np[b].numpy(),  # [T, Q]
                "confs":    confs_np[b].numpy(),     # [T, Q]
            })

    # Average losses
    avg_losses = {k: v / n_batches for k, v in totals.items()}
    
    metrics = evaluate(all_frame_preds, all_frame_gts, latency=0.0)

    # Overview plot — 4 clips side-by-side (existing)
    if sample_images:
        os.makedirs(save_dir, exist_ok=True)
        vis_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
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

        # Each clip contains one object/class, so any scoring heuristic
        # is meaningless — just pick a random clip from the batch.
        chosen = _random.randrange(len(temporal_clips))

        temp_path = os.path.join(save_dir, f"epoch_{epoch:03d}_temporal.png")
        visualize_temporal(
            clip=temporal_clips[chosen],
            conf_threshold=CONF_THRESHOLD,
            max_preds_shown=MAX_PREDS_CLIP,
            save_path=temp_path,
        )
        mlflow.log_artifact(temp_path, artifact_path="visuals")

    # Cleanup to prevent RAM growth across epochs
    all_frame_preds.clear()
    all_frame_gts.clear()
    _vis_res_gpu.clear()
    _vis_gt_boxes_gpu.clear()
    _vis_gt_classes_gpu.clear()
    _vis_boxes_gpu.clear()
    _vis_cls_ids_gpu.clear()
    _vis_confs_gpu.clear()
    temporal_clips.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_losses, metrics

# Main entry point
def main(config_path: str, resume: str | None = None, npz_dir_override: str | None = None) -> None:

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        print(f"[train] Loaded config from {config_path}")

    if npz_dir_override is not None:
        cfg["data"]["npz_dir"] = npz_dir_override
        print(f"[train] npz_dir overridden to: {npz_dir_override}")

    cfg_model  = cfg["model"]
    cfg_data   = cfg["data"]
    cfg_train  = cfg["training"]
    cfg_loss   = cfg["loss"]
    cfg_paths  = cfg["paths"]
    exp_name   = cfg.get("experiment_name", "experiment")

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}  |  Experiment: {exp_name}")

    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    torch.manual_seed(cfg_train.get("seed", 42))
    print(f"[train] Config: {cfg}")

    # mlflow — only rank 0 logs
    mlflow_port = os.environ.get("MLFLOW_PORT", "8501")
    mlflow.set_tracking_uri(f"http://localhost:{mlflow_port}")
    mlflow.set_experiment(exp_name)

    run_name    = cfg.get("run_name", "training_run")
    config_name = os.path.basename(config_path).replace(".yaml", "")

    mlflow.start_run(run_name=f"{run_name}_{config_name}")
    mlflow.set_tag("config_name", config_name)
    mlflow.log_dict(cfg, "config.yaml")
    mlflow.log_params({
        "lr":          cfg_train["lr"],
        "epochs":      cfg_train["epochs"],
        "clip_length": cfg_data["clip_length"],
        "batch_size":  cfg_train.get("batch_size", "unknown"),
    })
    mlflow.set_tag("config", yaml.dump(cfg, default_flow_style=False))

    # mixed prec
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Build model
    frame_h = cfg_data["frame_h"]
    frame_w = cfg_data["frame_w"]
    scales  = cfg_model["scales"]

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
        num_queries        = cfg_model.get("num_queries", 10),
    ).to(device)


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # Data loaders
    npz_dir     = cfg_data.get("npz_dir")
    clip_length = cfg_data["clip_length"]
    stride      = cfg_data.get("stride", clip_length)
    snap        = cfg_data.get("snap_to_iframe", True)

    # num_workers: spread CPUs across workers.  With 15 CPUs and 1 GPU, use 12
    # num_workers from config
    num_workers = cfg_train.get("num_workers", 12)

    train_loader, val_loader, test_loader = build_data_loaders(
            npz_dir=cfg_data["npz_dir"],
            clip_length=cfg_data["clip_length"],
            stride=cfg_data["stride"],
            snap_to_iframe=cfg_data["snap_to_iframe"],
            max_files=cfg_data.get("max_files"),
            max_files_per_class=cfg_data.get("max_files_per_class"),
            batch_size=cfg_train["batch_size"],
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            target_classes=cfg_data.get("target_classes", []),
            use_motionvectors=cfg_data.get("use_motionvectors", True),
            use_residuals=cfg_data.get("use_residuals", True),
            train_ratio=cfg_data.get("train_ratio", 0.8),
            val_ratio=cfg_data.get("val_ratio", 0.1),
            test_ratio=cfg_data.get("test_ratio", 0.1),
            prefetch_factor=cfg_train.get("prefetch_factor", 4),
            persistent_workers=cfg_train.get("persistent_workers", True),
            sequential_io=cfg_data.get("sequential_io", False),
        )

    # debug
    if cfg_train.get("debug_first_batch", False):
        batch = next(iter(train_loader))
        print({k: v.shape if isinstance(v, torch.Tensor) else "list"
           for k, v in batch.items()})

    # Safely get sizes even if val/test are empty lists
    val_size  = len(val_loader.dataset)  if hasattr(val_loader,  "dataset") else 0
    test_size = len(test_loader.dataset) if hasattr(test_loader, "dataset") else 0

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
        if warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs)
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif sched_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    # Checkpoint dir & config snapshot (rank 0 only)
    ckpt_dir = os.path.join(cfg_paths.get("checkpoint_dir", "checkpoints"), exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # Optionally resume
    start_epoch   = 0
    best_val_loss = float("inf")
    best_epoch    = 0

    if resume is not None:
        ckpt          = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[train] Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        train_losses = train_one_epoch(
            model, train_loader, optimizer, cfg_loss, cfg_train, device, scaler
        )

        # Validation and logging only on rank 0
        if val_loader:
            val_losses, val_metrics = validate(model, val_loader, cfg_loss, device, cfg_model["num_classes"], epoch)
        else:
            print("  [train] No validation data, skipping validate()")
            val_losses  = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
            val_metrics = None

        # Step scheduler
        if sched_type == "plateau":
            scheduler.step(val_losses["loss_total"])
        else:
            scheduler.step()

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
            "train_loss_cls":   train_losses["loss_cls"],
            "train_loss_l1":    train_losses["loss_l1"],
            "train_loss_giou":  train_losses["loss_giou"],
            "val_loss_total":   val_losses["loss_total"],
            "learning_rate":    lr_now,
        }

        if val_loader:
            metrics_to_log.update({
                "val_accuracy": val_metrics.accuracy,
                "val_mAP_50":   val_metrics.mAP_50,
                "val_IoU":      val_metrics.iou,
            })

        mem_info  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        metrics_to_log.update({
            "my_gpu/memory_used_MB":      mem_info.used // (1024 ** 2),
            "my_gpu/memory_total_MB":     mem_info.total // (1024 ** 2),
            "my_gpu/utilization_percent": util_info.gpu,
        })

        mlflow.log_metrics(metrics_to_log, step=epoch)

        if val_loader:
            print(
                f"| acc {val_metrics.accuracy:.3f} "
                f"| mAP50 {val_metrics.mAP_50:.3f} "
                f"| IoU {val_metrics.iou:.3f} "
            )

        print(f"| lr {lr_now:.2e} | {elapsed:.1f}s")

        # Save last checkpoint every epoch
        raw_model = model
        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(
            {
                "epoch":     epoch,
                "model":     raw_model.state_dict(),
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
            best_epoch    = epoch
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(
                {
                    "epoch":     epoch,
                    "model":     raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_loss":  best_val_loss,
                    "config":    cfg,
                },
                best_path,
            )
            print(f"  > New best val_loss {best_val_loss:.4f} — saved to {best_path}")
            mlflow.log_artifact(best_path, artifact_path="models")

        # End-of-epoch memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Early stopping
        patience = cfg_train.get("early_stopping_patience", 0)
        if patience > 0 and (epoch - best_epoch) >= patience:
            print(f"[train] Early stopping at epoch {epoch+1} — no improvement for {patience} epochs (best was epoch {best_epoch+1})")
            break

    print(f"\n[train] Done. Best val_loss: {best_val_loss:.4f}")
    mlflow.end_run()

    pynvml.nvmlShutdown()


# main loop where all is actiaveted
#could be paired with checks firstly
if __name__ == "__main__":
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