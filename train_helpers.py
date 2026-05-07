"""
train_helpers.py  —  loss computation for the compressed-domain object detector.

Kept separate from train.py so the loss logic is testable in isolation and
train.py stays focused on the training loop.

    giou_loss       generalised IoU loss between predicted and GT boxes
    canonical_xyxy  ensure xmin < xmax, ymin < ymax before computing losses
    compute_loss    full DETR-style set-prediction loss (vectorised, no loop)
"""

import torch
import torch.nn.functional as F


# Box utilities
def canonical_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Sort each box's coords so xmin ≤ xmax and ymin ≤ ymax.  Shape: [..., 4]."""
    return torch.stack([
        torch.minimum(boxes[..., 0], boxes[..., 1]),  # xmin
        torch.maximum(boxes[..., 0], boxes[..., 1]),  # xmax
        torch.minimum(boxes[..., 2], boxes[..., 3]),  # ymin
        torch.maximum(boxes[..., 2], boxes[..., 3]),  # ymax
    ], dim=-1)


def giou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU loss.  Returns 1 - GIoU, shape matches leading dims of inputs.
    Boxes in [xmin, xmax, ymin, ymax] format, normalised to [0, 1].
    Supports broadcast: pred [..., 4] and gt [..., 4].
    """
    p_x1, p_x2 = pred[..., 0], pred[..., 1]
    p_y1, p_y2 = pred[..., 2], pred[..., 3]
    g_x1, g_x2 = gt[..., 0],   gt[..., 1]
    g_y1, g_y2 = gt[..., 2],   gt[..., 3]

    # Intersection
    inter_w = (torch.min(p_x2, g_x2) - torch.max(p_x1, g_x1)).clamp(min=0)
    inter_h = (torch.min(p_y2, g_y2) - torch.max(p_y1, g_y1)).clamp(min=0)
    inter   = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_g = (g_x2 - g_x1).clamp(min=0) * (g_y2 - g_y1).clamp(min=0)
    union  = area_p + area_g - inter + 1e-7
    iou    = inter / union

    # Enclosing box
    enc_w = (torch.max(p_x2, g_x2) - torch.min(p_x1, g_x1)).clamp(min=0)
    enc_h = (torch.max(p_y2, g_y2) - torch.min(p_y1, g_y1)).clamp(min=0)
    enc   = enc_w * enc_h + 1e-7

    return 1.0 - (iou - (enc - union) / enc)


# Loss
def compute_loss(
    pred_boxes:   torch.Tensor,   # [B, T, Q, 4]
    pred_classes: torch.Tensor,   # [B, T, Q, num_classes]
    gt_boxes:     torch.Tensor,   # [B, T, 4]
    gt_classes:   torch.Tensor,   # [B, T]   (-1 = no object)
    cfg_loss:     dict,
) -> tuple[torch.Tensor, dict]:
    """
    Vectorised DETR-style set-prediction loss.

    Each (b, t) frame has at most one GT object, so Hungarian matching
    reduces to argmin over the query cost vector — no LSA library needed.
    All B*T frames are processed in one batched pass (no Python loop).

    Returns (total_loss, component_dict).
    component_dict keys: loss_total, loss_cls, loss_l1, loss_giou.
    """
    B, T, Q, C = pred_classes.shape
    no_obj      = C - 1
    BT          = B * T

    w_l1    = cfg_loss.get("bbox_l1_weight",   5.0)
    w_giou  = cfg_loss.get("bbox_giou_weight", 2.0)
    w_cls   = cfg_loss.get("class_weight",     1.0)
    w_noobj = cfg_loss.get("no_object_weight", 0.1)

    pred_boxes = canonical_xyxy(pred_boxes)
    probs      = torch.softmax(pred_classes, dim=-1)

    ce_weight           = pred_classes.new_ones(C)
    ce_weight[no_obj]   = w_noobj

    # Flatten (B, T) → BT
    p_boxes  = pred_boxes.reshape(BT, Q, 4)
    p_cls    = pred_classes.reshape(BT, Q, C)
    p_probs  = probs.reshape(BT, Q, C)
    g_boxes  = gt_boxes.reshape(BT, 4)
    g_cls    = gt_classes.reshape(BT)

    valid      = g_cls != -1                        # [BT] bool
    n_valid    = int(valid.sum().item())             # one sync — unavoidable

    # Default: all queries supervise as no-object
    targets = torch.full((BT, Q), no_obj, dtype=torch.long, device=p_cls.device)

    if n_valid > 0:
        vp_boxes = p_boxes[valid]                   # [V, Q, 4]
        vp_probs = p_probs[valid]                   # [V, Q, C]
        vg_boxes = g_boxes[valid]                   # [V, 4]
        vg_cls   = g_cls[valid].long()              # [V]

        # Cost per query against each frame's single GT  →  [V, Q]
        cost_cls  = -vp_probs[:, :, :no_obj].gather(
            2, vg_cls.view(-1, 1, 1).expand(-1, Q, 1)
        ).squeeze(-1)
        cost_l1   = (vp_boxes - vg_boxes.unsqueeze(1)).abs().sum(-1)
        cost_giou = giou_loss(vp_boxes, vg_boxes.unsqueeze(1))
        cost      = w_cls * cost_cls + w_l1 * cost_l1 + w_giou * cost_giou

        # With one GT per frame, argmin is exact Hungarian
        matched_q = cost.argmin(dim=1)              # [V]

        valid_idx = valid.nonzero(as_tuple=True)[0]
        targets[valid_idx, matched_q] = vg_cls

        matched_pred = vp_boxes.gather(
            1, matched_q.view(-1, 1, 1).expand(-1, 1, 4)
        ).squeeze(1)                                # [V, 4]

        loss_l1   = ((matched_pred - vg_boxes).abs().sum(-1) / 4.0).sum() / n_valid
        loss_giou = giou_loss(matched_pred, vg_boxes).sum() / n_valid
    else:
        loss_l1   = (pred_boxes * 0).sum()
        loss_giou = (pred_boxes * 0).sum()

    loss_cls = F.cross_entropy(
        p_cls.reshape(BT * Q, C),
        targets.reshape(BT * Q),
        weight=ce_weight,
        reduction="sum",
    ) / (BT * Q)

    total = w_cls * loss_cls + w_l1 * loss_l1 + w_giou * loss_giou
    return total, {
        "loss_total": total.detach().item(),
        "loss_cls":   (w_cls  * loss_cls).detach().item(),
        "loss_l1":    (w_l1   * loss_l1).detach().item(),
        "loss_giou":  (w_giou * loss_giou).detach().item(),
    }