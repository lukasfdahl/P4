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

# ── Only difference from train.py: import ResNet50Detector instead of ObjectDetector
from resnet50 import ResNet50Detector
from dataloader import build_data_loaders
from helpers import (
    clips_from_long_videos,
    save_checkpoint,
    load_checkpoint,
    xyxy_to_xywh,
    load_clips_from_npz_dir,
)
from eval_framwork import BoundingBox, Prediction, evaluate, ModelMetric

import mlflow
import os
from tqdm import tqdm
import pynvml


def giou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU loss for [xmin, xmax, ymin, ymax] normalised boxes.

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box of A and B.
    Loss = 1 - GIoU  (in [0, 2]).

    Shapes: [..., 4]  →  [...] scalar per box pair.
    """
    p_xmin, p_xmax = pred_boxes[..., 0], pred_boxes[..., 1]
    p_ymin, p_ymax = pred_boxes[..., 2], pred_boxes[..., 3]

    g_xmin, g_xmax = gt_boxes[..., 0], gt_boxes[..., 1]
    g_ymin, g_ymax = gt_boxes[..., 2], gt_boxes[..., 3]

    ix1 = torch.max(p_xmin, g_xmin)
    iy1 = torch.max(p_ymin, g_ymin)
    ix2 = torch.min(p_xmax, g_xmax)
    iy2 = torch.min(p_ymax, g_ymax)

    inter_w      = (ix2 - ix1).clamp(min=0.0)
    inter_h      = (iy2 - iy1).clamp(min=0.0)
    intersection = inter_w * inter_h

    area_pred = (p_xmax - p_xmin).clamp(min=0.0) * (p_ymax - p_ymin).clamp(min=0.0)
    area_gt   = (g_xmax - g_xmin).clamp(min=0.0) * (g_ymax - g_ymin).clamp(min=0.0)
    union     = area_pred + area_gt - intersection + 1e-7

    iou = intersection / union

    enc_x1 = torch.min(p_xmin, g_xmin)
    enc_y1 = torch.min(p_ymin, g_ymin)
    enc_x2 = torch.max(p_xmax, g_xmax)
    enc_y2 = torch.max(p_ymax, g_ymax)

    enc_area = ((enc_x2 - enc_x1).clamp(min=0.0) * (enc_y2 - enc_y1).clamp(min=0.0)) + 1e-7

    giou = iou - (enc_area - union) / enc_area
    return 1.0 - giou


def canonical_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Ensure [xmin, xmax, ymin, ymax] ordering.
    Prediction head uses sigmoid but does not guarantee xmin <= xmax or ymin <= ymax.
    """
    xmin = torch.minimum(boxes[..., 0], boxes[..., 1])
    xmax = torch.maximum(boxes[..., 0], boxes[..., 1])
    ymin = torch.minimum(boxes[..., 2], boxes[..., 3])
    ymax = torch.maximum(boxes[..., 2], boxes[..., 3])
    return torch.stack([xmin, xmax, ymin, ymax], dim=-1)


def compute_loss(
    pred_boxes:   torch.Tensor,   # [B, num_queries, 4]
    pred_classes: torch.Tensor,   # [B, num_queries, num_classes + 1]
    gt_boxes:     torch.Tensor,   # [B, T, 4]
    gt_classes:   torch.Tensor,   # [B, T]
    cfg_loss:     dict,
) -> tuple[torch.Tensor, dict]:

    B, num_queries, num_classes = pred_classes.shape
    no_object_class = num_classes - 1
    pred_boxes = canonical_xyxy(pred_boxes)

    w_l1    = cfg_loss.get("bbox_l1_weight",   5.0)
    w_giou  = cfg_loss.get("bbox_giou_weight", 2.0)
    w_cls   = cfg_loss.get("class_weight",     1.0)
    w_noobj = cfg_loss.get("no_object_weight", 0.1)

    probs = torch.softmax(pred_classes, dim=-1)

    ce_weight = pred_classes.new_ones(num_classes)
    ce_weight[no_object_class] = w_noobj

    total_l1   = pred_boxes.new_tensor(0.0)
    total_giou = pred_boxes.new_tensor(0.0)
    total_cls  = pred_boxes.new_tensor(0.0)
    valid_frames  = 0
    total_queries = 0

    for b in range(B):
        valid_mask       = gt_classes[b] != -1
        valid_gt_boxes   = gt_boxes[b][valid_mask]
        valid_gt_classes = gt_classes[b][valid_mask].long()

        target_classes = torch.full(
            (num_queries,),
            fill_value=no_object_class,
            dtype=torch.long,
            device=pred_classes.device,
        )

        num_gt = valid_gt_boxes.shape[0]
        if num_gt > 0:
            out_prob   = probs[b][:, :no_object_class]
            cost_class = -out_prob[:, valid_gt_classes]
            cost_l1    = torch.cdist(pred_boxes[b], valid_gt_boxes, p=1)
            cost_giou  = giou_loss(
                pred_boxes[b].unsqueeze(1),
                valid_gt_boxes.unsqueeze(0),
            )

            C          = (w_cls * cost_class) + (w_l1 * cost_l1) + (w_giou * cost_giou)
            assignment = gpu_lsa(C.unsqueeze(0)).squeeze(0)

            matched_mask = assignment >= 0
            row_t = matched_mask.nonzero(as_tuple=True)[0].to(dtype=torch.long)
            col_t = assignment[matched_mask].to(dtype=torch.long)

            matched_boxes    = pred_boxes[b][row_t]
            matched_gt_boxes = valid_gt_boxes[col_t]
            matched_gt_cls   = valid_gt_classes[col_t]

            M = row_t.shape[0]
            if M > 0:
                target_classes[row_t] = matched_gt_cls
                total_l1   += F.l1_loss(matched_boxes, matched_gt_boxes, reduction="sum") / 4.0
                total_giou += giou_loss(matched_boxes, matched_gt_boxes).sum()
                valid_frames += M

        total_cls += F.cross_entropy(
            pred_classes[b],
            target_classes,
            weight=ce_weight,
            reduction="sum",
        )
        total_queries += num_queries

    if total_queries == 0:
        zero = (pred_boxes * 0).sum() + (pred_classes * 0).sum()
        return zero, {
            "loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0
        }

    if valid_frames > 0:
        total_l1   = total_l1   / valid_frames
        total_giou = total_giou / valid_frames
    else:
        total_l1   = (pred_boxes * 0).sum()
        total_giou = (pred_boxes * 0).sum()

    total_cls = total_cls / total_queries
    total     = (w_cls * total_cls) + (w_l1 * total_l1) + (w_giou * total_giou)

    return total, {
        "loss_total": total.detach().item(),
        "loss_cls":   (w_cls * total_cls).detach().item(),
        "loss_l1":    (w_l1 * total_l1).detach().item(),
        "loss_giou":  (w_giou * total_giou).detach().item(),
    }


def train_one_epoch(
    model:     ResNet50Detector,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg_loss:  dict,
    cfg_train: dict,
    device:    torch.device,
    scaler:    torch.cuda.amp.GradScaler,
) -> dict:

    model.train()
    totals    = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for batch in pbar:

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
        pbar.set_postfix(loss=f"{parts['loss_total']:.4f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)


@torch.no_grad()
def validate(
    model:       ResNet50Detector,
    loader:      torch.utils.data.DataLoader,
    cfg_loss:    dict,
    device:      torch.device,
    num_classes: int,
    epoch:       int,
    save_dir:    str = "visuals",
) -> tuple[dict, ModelMetric]:

    model.eval()
    totals    = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
    n_batches = 0

    all_frame_preds = []
    all_frame_gts   = []
    sample_images   = []

    _vis_res_gpu:        list[torch.Tensor] = []
    _vis_gt_boxes_gpu:   list[torch.Tensor] = []
    _vis_gt_classes_gpu: list[torch.Tensor] = []
    _vis_boxes_gpu:      list[torch.Tensor] = []
    _vis_cls_ids_gpu:    list[torch.Tensor] = []
    _vis_confs_gpu:      list[torch.Tensor] = []

    CONF_THRESHOLD = 0.5
    MAX_PREDS_CLIP = 10

    temporal_clips: list = []

    for i, batch in enumerate(tqdm(loader, desc="  val  ", leave=False, unit="batch")):
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

        B, Q, _ = pred_boxes.shape
        probs   = torch.softmax(pred_logits, dim=-1)[..., :num_classes]

        boxes_eval = torch.stack([
            torch.minimum(pred_boxes[..., 0], pred_boxes[..., 1]),
            torch.maximum(pred_boxes[..., 0], pred_boxes[..., 1]),
            torch.minimum(pred_boxes[..., 2], pred_boxes[..., 3]),
            torch.maximum(pred_boxes[..., 2], pred_boxes[..., 3]),
        ], dim=-1)

        confs_batch, cls_ids_batch = torch.max(probs, dim=-1)

        for b in range(B):
            frame_preds = []
            frame_gts   = []

            confs   = confs_batch[b]
            cls_ids = cls_ids_batch[b]

            mask         = confs >= CONF_THRESHOLD
            cand_confs   = confs[mask]
            cand_cls     = cls_ids[mask]
            cand_boxes   = boxes_eval[b][mask]
            topk         = min(MAX_PREDS_CLIP, cand_confs.shape[0])
            if topk > 0:
                topk_idx       = torch.argsort(cand_confs, descending=True)[:topk]
                cand_confs_np  = cand_confs[topk_idx].cpu().numpy()
                cand_cls_np    = cand_cls[topk_idx].cpu().numpy()
                cand_boxes_np  = cand_boxes[topk_idx].cpu().numpy()
                for k_idx in range(topk):
                    frame_preds.append(Prediction(
                        xmin=float(cand_boxes_np[k_idx, 0]),
                        xmax=float(cand_boxes_np[k_idx, 1]),
                        ymin=float(cand_boxes_np[k_idx, 2]),
                        ymax=float(cand_boxes_np[k_idx, 3]),
                        class_id=int(cand_cls_np[k_idx]),
                        confidence=float(cand_confs_np[k_idx]),
                    ))

            valid_mask_gt = gt_classes[b] != -1
            valid_gt_b    = gt_boxes[b][valid_mask_gt]
            valid_cls_b   = gt_classes[b][valid_mask_gt]
            if valid_gt_b.shape[0] > 0:
                valid_gt_np  = valid_gt_b.cpu().numpy()
                valid_cls_np = valid_cls_b.cpu().numpy()
                for m in range(valid_gt_np.shape[0]):
                    frame_gts.append(BoundingBox(
                        xmin=float(valid_gt_np[m, 0]),
                        xmax=float(valid_gt_np[m, 1]),
                        ymin=float(valid_gt_np[m, 2]),
                        ymax=float(valid_gt_np[m, 3]),
                        class_id=int(valid_cls_np[m]),
                    ))

            all_frame_preds.append(frame_preds)
            all_frame_gts.append(frame_gts)

        # Collect visuals from batch 0 only
        if i == 0 and res is not None:
            for b in range(min(B, 4)):
                sample_images.append(res[b, 0])
                _vis_res_gpu.append(res[b])
                _vis_gt_boxes_gpu.append(gt_boxes[b])
                _vis_gt_classes_gpu.append(gt_classes[b])
                _vis_boxes_gpu.append(boxes_eval[b])
                _vis_cls_ids_gpu.append(cls_ids_batch[b])
                _vis_confs_gpu.append(confs_batch[b])

    if sample_images:
        sample_images_np = torch.stack(sample_images).cpu().permute(0, 2, 3, 1).numpy()
        sample_images    = [sample_images_np[k] for k in range(sample_images_np.shape[0])]

    if _vis_res_gpu:
        res_np      = torch.stack(_vis_res_gpu).cpu()
        gt_boxes_np = torch.stack(_vis_gt_boxes_gpu).cpu()
        gt_cls_np   = torch.stack(_vis_gt_classes_gpu).cpu()
        boxes_np    = torch.stack(_vis_boxes_gpu).cpu()
        cls_ids_np  = torch.stack(_vis_cls_ids_gpu).cpu()
        confs_np    = torch.stack(_vis_confs_gpu).cpu()

        for b in range(res_np.shape[0]):
            clip_images = [res_np[b, t].permute(1, 2, 0).numpy() for t in range(res_np.shape[1])]
            gt_per_t = []
            for t in range(gt_cls_np.shape[1]):
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
                "boxes":    boxes_np[b].numpy(),
                "cls_ids":  cls_ids_np[b].numpy(),
                "confs":    confs_np[b].numpy(),
            })

    avg_losses = {k: v / n_batches for k, v in totals.items()}
    metrics    = evaluate(all_frame_preds, all_frame_gts, latency=0.0)

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

    if temporal_clips:
        from eval_framwork import visualize_temporal
        import random as _random
        chosen    = _random.randrange(len(temporal_clips))
        temp_path = os.path.join(save_dir, f"epoch_{epoch:03d}_temporal.png")
        visualize_temporal(
            clip=temporal_clips[chosen],
            conf_threshold=CONF_THRESHOLD,
            max_preds_shown=MAX_PREDS_CLIP,
            save_path=temp_path,
        )
        mlflow.log_artifact(temp_path, artifact_path="visuals")

    return avg_losses, metrics


def main(config_path: str, resume: str | None = None, npz_dir_override: str | None = None) -> None:

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

    pynvml.nvmlInit()
    physical_gpu_id = int(gpu_id.split(",")[0]) if gpu_id.strip() else 0
    gpu_handle      = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_id)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        print(f"[train] Loaded config from {config_path}")

    if npz_dir_override is not None:
        cfg["data"]["npz_dir"] = npz_dir_override
        print(f"[train] npz_dir overridden to: {npz_dir_override}")

    cfg_model = cfg["model"]
    cfg_data  = cfg["data"]
    cfg_train = cfg["training"]
    cfg_loss  = cfg["loss"]
    cfg_paths = cfg["paths"]
    exp_name  = cfg.get("experiment_name", "experiment")

    torch.manual_seed(cfg_train.get("seed", 42))
    print(f"[train] Config: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}  |  Experiment: {exp_name}")

    mlflow_port = os.environ.get("MLFLOW_PORT", "8501")
    mlflow.set_tracking_uri(f"http://localhost:{mlflow_port}")
    mlflow.set_experiment(exp_name)

    run_name    = cfg.get("run_name", "training_run")
    config_name = os.path.basename(config_path).replace(".yaml", "")
    mlflow.start_run(run_name=f"{run_name}_{config_name}")

    mlflow.set_tag("config_name", config_name)
    mlflow.log_dict(cfg, "config.yaml")
    mlflow.log_params({
        "lr":           cfg_train["lr"],
        "epochs":       cfg_train["epochs"],
        "clip_length":  cfg_data["clip_length"],
        "batch_size":   cfg_train.get("batch_size", "unknown"),
        "input_mode":   cfg_model.get("input_mode", "mv_concat"),   # extra resnet param
        "pretrained":   cfg_model.get("pretrained", False),          # extra resnet param
    })
    mlflow.set_tag("config", yaml.dump(cfg, default_flow_style=False))
    mlflow.set_tag("gpu_id", gpu_id)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Build ResNet50Detector from config ────────────────────────────────────
    model = ResNet50Detector(
        num_classes   = cfg_model["num_classes"],
        num_queries   = cfg_model.get("num_queries",   10),
        hidden_dim    = cfg_model.get("hidden_dim",    256),
        input_mode    = cfg_model.get("input_mode",    "mv_concat"),
        pretrained    = cfg_model.get("pretrained",    False),
        base_mv_scale = cfg_model.get("base_mv_scale", 16),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] ResNet50Detector — input_mode={model.input_mode}  "
          f"pretrained={cfg_model.get('pretrained', False)}  "
          f"params={n_params:,}")

    # Data loaders
    train_loader, val_loader, test_loader = build_data_loaders(
        npz_dir             = cfg_data["npz_dir"],
        clip_length         = cfg_data["clip_length"],
        stride              = cfg_data["stride"],
        snap_to_iframe      = cfg_data["snap_to_iframe"],
        max_files           = cfg_data.get("max_files"),
        max_files_per_class = cfg_data.get("max_files_per_class"),
        batch_size          = cfg_train["batch_size"],
        num_workers         = cfg_train.get("num_workers", 4),
        pin_memory          = (device.type == "cuda"),
        target_classes      = cfg_data.get("target_classes", []),
        use_motionvectors   = cfg_data.get("use_motionvectors", True),
        use_residuals       = cfg_data.get("use_residuals",    True),
        train_ratio         = cfg_data.get("train_ratio",  0.8),
        val_ratio           = cfg_data.get("val_ratio",    0.1),
        test_ratio          = cfg_data.get("test_ratio",   0.1),
        prefetch_factor     = cfg_train.get("prefetch_factor",    2),
        persistent_workers  = cfg_train.get("persistent_workers", True),
    )

    if cfg_train.get("debug_first_batch", False):
        batch = next(iter(train_loader))
        print({k: v.shape if isinstance(v, torch.Tensor) else "list"
               for k, v in batch.items()})

    val_size  = len(val_loader.dataset)  if hasattr(val_loader,  "dataset") else 0
    test_size = len(test_loader.dataset) if hasattr(test_loader, "dataset") else 0
    print(f"  Split sizes  |  train={len(train_loader.dataset)}"
          f"  val={val_size}  test={test_size}")

    optimizer     = AdamW(model.parameters(),
                          lr=cfg_train["lr"],
                          weight_decay=cfg_train.get("weight_decay", 1e-4))
    epochs        = cfg_train["epochs"]
    warmup_epochs = cfg_train.get("warmup_epochs", 3)
    sched_type    = cfg_train.get("scheduler", "cosine")

    if sched_type == "cosine":
        if warmup_epochs > 0:
            warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                                    total_iters=warmup_epochs)
            cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
            scheduler    = SequentialLR(optimizer,
                                        schedulers=[warmup_sched, cosine_sched],
                                        milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif sched_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

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

    # Training loop
    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        train_losses = train_one_epoch(
            model, train_loader, optimizer, cfg_loss, cfg_train, device, scaler
        )

        if val_loader:
            val_losses, val_metrics = validate(
                model, val_loader, cfg_loss, device, cfg_model["num_classes"], epoch
            )
        else:
            print("  [train] No validation data, skipping validate()")
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
            print(
                f"| acc {val_metrics.accuracy:.3f} "
                f"| mAP50 {val_metrics.mAP_50:.3f} "
                f"| IoU {val_metrics.iou:.3f} "
            )

        print(f"| lr {lr_now:.2e} | {elapsed:.1f}s")

        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "val_loss": val_losses["loss_total"], "config": cfg},
            last_path,
        )

        if val_losses["loss_total"] < best_val_loss:
            best_val_loss = val_losses["loss_total"]
            best_path     = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "val_loss": best_val_loss, "config": cfg},
                best_path,
            )
            print(f"  ↳ New best val_loss {best_val_loss:.4f} — saved to {best_path}")
            mlflow.log_artifact(best_path, artifact_path="models")

    print(f"\n[train] Done. Best val_loss: {best_val_loss:.4f}")
    mlflow.end_run()
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config_resnet50_benchmark.yaml")
    parser.add_argument("--npz-dir", default=None,
                        help="Override data.npz_dir from config (e.g. local scratch path)")
    args = parser.parse_args()

    with open(args.config) as f:
        _cfg = yaml.safe_load(f)

    main(
        config_path      = args.config,
        resume           = _cfg.get("resume", None),
        npz_dir_override = args.npz_dir,
    )


# mlflow server --host 127.0.0.1 --port 8080
# lsof -ti:8080 | xargs kill -9