"""
train_fasterrcnn.py  –  Training script for FasterRCNNDetector baseline.

Intentionally mirrors train_resnet50.py as closely as possible so the only
variable between experiments is the model, not the training procedure.

Key difference vs train.py / train_resnet50.py:
  - Imports FasterRCNNDetector from faster_rcnn.py
  - No motion-vector input (use_motionvectors: false in config)
  - Batch size is smaller (256 vs 1024) because Faster R-CNN is heavier

The loss function, eval framework, and dataloader are identical to all other
experiments — this is intentional for a fair comparison.
"""

import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from torch_linear_assignment import batch_linear_assignment as gpu_lsa

from faster_rcnn import FasterRCNNDetector
from dataloader import build_data_loaders
from helpers import save_checkpoint, load_checkpoint
from eval_framwork import BoundingBox, Prediction, evaluate, ModelMetric

import mlflow
from tqdm import tqdm
import pynvml


# ── Loss (identical to train.py) ──────────────────────────────────────────────

def giou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    p_xmin, p_xmax = pred_boxes[..., 0], pred_boxes[..., 1]
    p_ymin, p_ymax = pred_boxes[..., 2], pred_boxes[..., 3]
    g_xmin, g_xmax = gt_boxes[..., 0], gt_boxes[..., 1]
    g_ymin, g_ymax = gt_boxes[..., 2], gt_boxes[..., 3]

    ix1 = torch.max(p_xmin, g_xmin);  iy1 = torch.max(p_ymin, g_ymin)
    ix2 = torch.min(p_xmax, g_xmax);  iy2 = torch.min(p_ymax, g_ymax)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    area_p = (p_xmax - p_xmin).clamp(0) * (p_ymax - p_ymin).clamp(0)
    area_g = (g_xmax - g_xmin).clamp(0) * (g_ymax - g_ymin).clamp(0)
    union  = area_p + area_g - inter + 1e-7
    iou    = inter / union

    enc_x1 = torch.min(p_xmin, g_xmin);  enc_y1 = torch.min(p_ymin, g_ymin)
    enc_x2 = torch.max(p_xmax, g_xmax);  enc_y2 = torch.max(p_ymax, g_ymax)
    enc    = ((enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)) + 1e-7

    return 1.0 - (iou - (enc - union) / enc)


def canonical_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        torch.minimum(boxes[..., 0], boxes[..., 1]),
        torch.maximum(boxes[..., 0], boxes[..., 1]),
        torch.minimum(boxes[..., 2], boxes[..., 3]),
        torch.maximum(boxes[..., 2], boxes[..., 3]),
    ], dim=-1)


def compute_loss(pred_boxes, pred_classes, gt_boxes, gt_classes, cfg_loss):
    B, Q, num_classes = pred_classes.shape
    no_obj  = num_classes - 1
    pred_boxes = canonical_xyxy(pred_boxes)

    w_l1    = cfg_loss.get("bbox_l1_weight",   5.0)
    w_giou  = cfg_loss.get("bbox_giou_weight", 2.0)
    w_cls   = cfg_loss.get("class_weight",     1.0)
    w_noobj = cfg_loss.get("no_object_weight", 0.1)

    probs     = torch.softmax(pred_classes, dim=-1)
    ce_weight = pred_classes.new_ones(num_classes)
    ce_weight[no_obj] = w_noobj

    target_classes = torch.full((B, Q), no_obj, dtype=torch.long, device=pred_classes.device)
    total_l1 = pred_boxes.new_tensor(0.0)
    total_giou = pred_boxes.new_tensor(0.0)
    total_matched = 0

    for b in range(B):
        valid = gt_classes[b] != -1
        vgt_b = gt_boxes[b][valid];   vgt_c = gt_classes[b][valid].long()
        if vgt_b.shape[0] == 0:
            continue
        cost = (
            w_cls  * (-probs[b, :, :no_obj][:, vgt_c]) +
            w_l1   * torch.cdist(pred_boxes[b], vgt_b, p=1) +
            w_giou * giou_loss(pred_boxes[b].unsqueeze(1), vgt_b.unsqueeze(0))
        )
        assignment = gpu_lsa(cost.unsqueeze(0)).squeeze(0)
        matched = assignment >= 0
        rows = matched.nonzero(as_tuple=True)[0]
        cols = assignment[matched].long()
        if rows.shape[0] > 0:
            target_classes[b, rows] = vgt_c[cols]
            total_l1   += F.l1_loss(pred_boxes[b][rows], vgt_b[cols], reduction="sum") / 4.0
            total_giou += giou_loss(pred_boxes[b][rows], vgt_b[cols]).sum()
            total_matched += rows.shape[0]

    total_cls = F.cross_entropy(
        pred_classes.view(B * Q, num_classes),
        target_classes.view(B * Q),
        weight=ce_weight, reduction="sum",
    ) / (B * Q)

    if total_matched > 0:
        total_l1   = total_l1   / total_matched
        total_giou = total_giou / total_matched
    else:
        total_l1   = (pred_boxes * 0).sum()
        total_giou = (pred_boxes * 0).sum()

    total = w_cls * total_cls + w_l1 * total_l1 + w_giou * total_giou
    return total, {
        "loss_total": total.detach().item(),
        "loss_cls":   (w_cls  * total_cls).detach().item(),
        "loss_l1":    (w_l1   * total_l1).detach().item(),
        "loss_giou":  (w_giou * total_giou).detach().item(),
    }


# ── Train / validate ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, cfg_loss, cfg_train, device, scaler):
    model.train()
    totals    = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="  train", leave=False, unit="batch"):
        gt_boxes    = batch["boxes"].to(device, non_blocking=True)
        gt_classes  = batch["true_class"].to(device, non_blocking=True)
        mv          = batch.get("motion_vectors")
        res         = batch.get("residuals")
        iframe_mask = batch["iframe_mask"].to(device, non_blocking=True)
        if mv  is not None: mv  = mv.to(device, non_blocking=True)
        if res is not None: res = res.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_boxes, pred_classes = model(mv, res, iframe_mask)
            loss, parts = compute_loss(pred_boxes, pred_classes, gt_boxes, gt_classes, cfg_loss)

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            scaler.update()

        for k, v in parts.items():
            totals[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, cfg_loss, device, num_classes, epoch, save_dir="visuals"):
    model.eval()
    totals    = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
    n_batches = 0
    all_preds = []
    all_gts   = []

    CONF = 0.05   # lower than ObjectDetector — Faster R-CNN scores are already thresholded
    Q    = 10

    for batch in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
        gt_boxes    = batch["boxes"].to(device, non_blocking=True)
        gt_classes  = batch["true_class"].to(device, non_blocking=True)
        mv          = batch.get("motion_vectors")
        res         = batch.get("residuals")
        iframe_mask = batch["iframe_mask"].to(device, non_blocking=True)
        if mv  is not None: mv  = mv.to(device, non_blocking=True)
        if res is not None: res = res.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_boxes, pred_logits = model(mv, res, iframe_mask)
            loss, parts = compute_loss(pred_boxes, pred_logits, gt_boxes, gt_classes, cfg_loss)

        for k, v in parts.items():
            totals[k] += v
        n_batches += 1

        B = pred_boxes.shape[0]
        probs = torch.softmax(pred_logits, dim=-1)[..., :num_classes]
        confs, cls_ids = torch.max(probs, dim=-1)

        boxes_eval = torch.stack([
            torch.minimum(pred_boxes[..., 0], pred_boxes[..., 1]),
            torch.maximum(pred_boxes[..., 0], pred_boxes[..., 1]),
            torch.minimum(pred_boxes[..., 2], pred_boxes[..., 3]),
            torch.maximum(pred_boxes[..., 2], pred_boxes[..., 3]),
        ], dim=-1)

        for b in range(B):
            frame_preds = []
            mask = confs[b] >= CONF
            if mask.any():
                cb = confs[b][mask].cpu().numpy()
                lb = cls_ids[b][mask].cpu().numpy()
                bb = boxes_eval[b][mask].cpu().numpy()
                for i in range(len(cb)):
                    frame_preds.append(Prediction(
                        xmin=float(bb[i, 0]), xmax=float(bb[i, 1]),
                        ymin=float(bb[i, 2]), ymax=float(bb[i, 3]),
                        class_id=int(lb[i]), confidence=float(cb[i]),
                    ))

            frame_gts = []
            valid_gt  = gt_classes[b] != -1
            if valid_gt.any():
                vb = gt_boxes[b][valid_gt].cpu().numpy()
                vc = gt_classes[b][valid_gt].cpu().numpy()
                for i in range(len(vc)):
                    frame_gts.append(BoundingBox(
                        xmin=float(vb[i, 0]), xmax=float(vb[i, 1]),
                        ymin=float(vb[i, 2]), ymax=float(vb[i, 3]),
                        class_id=int(vc[i]),
                    ))

            all_preds.append(frame_preds)
            all_gts.append(frame_gts)

    avg_losses = {k: v / max(n_batches, 1) for k, v in totals.items()}
    metrics    = evaluate(all_preds, all_gts, latency=0.0)
    return avg_losses, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str, resume: str | None = None, npz_dir_override: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"[train] Loaded config from {config_path}")

    if npz_dir_override is not None:
        cfg["data"]["npz_dir"] = npz_dir_override

    cfg_model = cfg["model"]
    cfg_data  = cfg["data"]
    cfg_train = cfg["training"]
    cfg_loss  = cfg["loss"]
    cfg_paths = cfg["paths"]
    exp_name  = cfg.get("experiment_name", "fasterrcnn_benchmark")

    torch.manual_seed(cfg_train.get("seed", 42))

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}  |  Experiment: {exp_name}")

    pynvml.nvmlInit()
    gpu_id     = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    mlflow_port = os.environ.get("MLFLOW_PORT", "8501")
    mlflow.set_tracking_uri(f"http://localhost:{mlflow_port}")
    mlflow.set_experiment(exp_name)

    run_name    = cfg.get("run_name", "fasterrcnn_run")
    config_name = os.path.basename(config_path).replace(".yaml", "")
    mlflow.start_run(run_name=f"{run_name}_{config_name}")
    mlflow.set_tag("config_name", config_name)
    mlflow.log_dict(cfg, "config.yaml")
    mlflow.log_params({
        "lr":         cfg_train["lr"],
        "epochs":     cfg_train["epochs"],
        "batch_size": cfg_train.get("batch_size", "unknown"),
        "pretrained": cfg_model.get("pretrained", False),
    })
    mlflow.set_tag("config", yaml.dump(cfg, default_flow_style=False))

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Build model
    model = FasterRCNNDetector(
        num_classes = cfg_model["num_classes"],
        num_queries = cfg_model.get("num_queries", 10),
        pretrained  = cfg_model.get("pretrained",  False),
        min_score   = cfg_model.get("min_score",   0.05),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] FasterRCNNDetector — pretrained={cfg_model.get('pretrained', False)}  "
          f"params={n_params:,}")

    train_loader, val_loader, test_loader = build_data_loaders(
        npz_dir             = cfg_data["npz_dir"],
        clip_length         = cfg_data["clip_length"],
        stride              = cfg_data["stride"],
        snap_to_iframe      = cfg_data["snap_to_iframe"],
        max_files           = cfg_data.get("max_files"),
        max_files_per_class = cfg_data.get("max_files_per_class"),
        batch_size          = cfg_train["batch_size"],
        num_workers         = cfg_train.get("num_workers", 12),
        pin_memory          = (device.type == "cuda"),
        target_classes      = cfg_data.get("target_classes", []),
        use_motionvectors   = cfg_data.get("use_motionvectors", False),
        use_residuals       = cfg_data.get("use_residuals",    True),
        train_ratio         = cfg_data.get("train_ratio", 0.7),
        val_ratio           = cfg_data.get("val_ratio",   0.3),
        test_ratio          = cfg_data.get("test_ratio",  0.0),
        prefetch_factor     = cfg_train.get("prefetch_factor",    4),
        persistent_workers  = cfg_train.get("persistent_workers", True),
    )

    val_size = len(val_loader.dataset) if hasattr(val_loader, "dataset") else 0
    print(f"  Split sizes  |  train={len(train_loader.dataset)}  val={val_size}")

    optimizer     = AdamW(model.parameters(), lr=cfg_train["lr"],
                          weight_decay=cfg_train.get("weight_decay", 1e-4))
    epochs        = cfg_train["epochs"]
    warmup_epochs = cfg_train.get("warmup_epochs", 3)
    sched_type    = cfg_train.get("scheduler", "cosine")

    if sched_type == "cosine":
        if warmup_epochs > 0:
            scheduler = SequentialLR(optimizer, schedulers=[
                LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs)),
            ], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif sched_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")

    ckpt_dir = os.path.join(cfg_paths.get("checkpoint_dir", "checkpoints"), exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    start_epoch   = 0
    best_val_loss = float("inf")

    if resume is not None:
        ckpt          = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[train] Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        train_losses = train_one_epoch(model, train_loader, optimizer, cfg_loss, cfg_train, device, scaler)

        if val_loader:
            val_losses, val_metrics = validate(model, val_loader, cfg_loss, device, cfg_model["num_classes"], epoch)
        else:
            val_losses  = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
            val_metrics = None

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
            "my_gpu/memory_used_MB":      mem_info.used  // (1024 ** 2),
            "my_gpu/memory_total_MB":     mem_info.total // (1024 ** 2),
            "my_gpu/utilization_percent": util_info.gpu,
        })
        mlflow.log_metrics(metrics_to_log, step=epoch)

        if val_loader:
            print(f"| acc {val_metrics.accuracy:.3f} | mAP50 {val_metrics.mAP_50:.3f} | IoU {val_metrics.iou:.3f} ")
        print(f"| lr {lr_now:.2e} | {elapsed:.1f}s")

        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "val_loss": val_losses["loss_total"], "config": cfg}, last_path)

        if val_losses["loss_total"] < best_val_loss:
            best_val_loss = val_losses["loss_total"]
            best_path     = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                             "val_loss": best_val_loss, "config": cfg}, best_path)
            print(f"  ↳ New best val_loss {best_val_loss:.4f} — saved to {best_path}")
            mlflow.log_artifact(best_path, artifact_path="models")

    print(f"\n[train] Done. Best val_loss: {best_val_loss:.4f}")
    mlflow.end_run()
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config_fasterrcnn_benchmark.yaml")
    parser.add_argument("--npz-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        _cfg = yaml.safe_load(f)

    main(config_path=args.config, resume=_cfg.get("resume"), npz_dir_override=args.npz_dir)