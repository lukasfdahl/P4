from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# Data containers
@dataclass
class BoundingBox:
    """
    Ground-truth bounding box.

    All coordinates are normalised to [0, 1] and stored as:
        xmin, xmax  – left and right edges
        ymin, ymax  – top and bottom edges

    This matches Frame.true_bounding_box in data_classes.py exactly.
    """
    xmin:     float
    xmax:     float
    ymin:     float
    ymax:     float
    class_id: int


@dataclass
class Prediction(BoundingBox):
    """
    Model prediction.  Inherits xmin/xmax/ymin/ymax/class_id from BoundingBox
    and adds a confidence score.
    """
    confidence: float   # 0–1, higher = more confident


@dataclass
class ModelMetric:
    accuracy:           float   # TP / (TP + FN) across all frames
    iou:                float   # mean IoU over matched TP pairs only
    mAP_50:             float   # macro-averaged AP per class at IoU 0.50
    mAP_95:             float   # single-threshold AP at IoU 0.95
    weighted_precision: float   # per-class precision at IoU 0.5, weighted by GT count
    latency:            float   # seconds per frame, passed in from caller

    def compare(self, other: "ModelMetric") -> Dict[str, float]:
        """
        Return percentage difference (self vs other) for every metric.
        Positive = self is higher.  Used to compare model variants.
        """
        def _pct(a: float, b: float) -> float:
            if b == 0.0:
                return float("inf") if a != 0.0 else 0.0
            return (a - b) / b * 100.0

        return {
            "accuracy_diff_pct":  _pct(self.accuracy,           other.accuracy),
            "iou_diff_pct":       _pct(self.iou,                other.iou),
            "mAP_50_diff_pct":    _pct(self.mAP_50,             other.mAP_50),
            "mAP_95_diff_pct":    _pct(self.mAP_95,             other.mAP_95),
            "precision_diff_pct": _pct(self.weighted_precision,  other.weighted_precision),
            "latency_diff_pct":   _pct(self.latency,             other.latency),
        }

    def __repr__(self) -> str:
        return (
            f"ModelMetric(\n"
            f"  accuracy           = {self.accuracy:.4f}\n"
            f"  iou                = {self.iou:.4f}\n"
            f"  mAP_50             = {self.mAP_50:.4f}\n"
            f"  mAP_95             = {self.mAP_95:.4f}\n"
            f"  weighted_precision = {self.weighted_precision:.4f}\n"
            f"  latency            = {self.latency:.4f}s/frame\n"
            f")"
        )


# IoU
def _compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Intersection-over-Union for two [xmin, xmax, ymin, ymax] boxes."""
    ix1 = max(box_a.xmin, box_b.xmin)
    iy1 = max(box_a.ymin, box_b.ymin)
    ix2 = min(box_a.xmax, box_b.xmax)
    iy2 = min(box_a.ymax, box_b.ymax)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin)
    area_b = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin)
    union  = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


# Matching

def _match_predictions(
    predictions:   List[Prediction],
    ground_truth:  List[BoundingBox],
    iou_threshold: float,
) -> Tuple[List[Tuple[Prediction, BoundingBox, float]], List[BoundingBox]]:
    """
    Greedy match: predictions sorted by descending confidence, each GT matched once.

    Returns (matched_pairs_with_iou, unmatched_gt).
    """
    matched:      List[Tuple[Prediction, BoundingBox, float]] = []
    unmatched_gt: List[BoundingBox]                           = list(ground_truth)

    for pred in sorted(predictions, key=lambda p: p.confidence, reverse=True):
        best_iou = 0.0
        best_gt  = None

        for gt in unmatched_gt:
            if gt.class_id != pred.class_id:
                continue
            iou = _compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt  = gt

        if best_gt is not None and best_iou >= iou_threshold:
            matched.append((pred, best_gt, best_iou))
            unmatched_gt.remove(best_gt)

    return matched, unmatched_gt


# Average Precision 

def _compute_ap_for_class(
    predictions:   List[Prediction],
    ground_truth:  List[BoundingBox],
    iou_threshold: float,
    class_id:      int,
) -> float:
    class_preds = [p for p in predictions if p.class_id == class_id]
    class_gts   = [g for g in ground_truth if g.class_id == class_id]

    if not class_gts or not class_preds:
        return 0.0

    class_preds = sorted(class_preds, key=lambda p: p.confidence, reverse=True)

    matched_gt_indices: set[int] = set()
    tp_list: List[int] = []
    fp_list: List[int] = []

    for pred in class_preds:
        best_iou = 0.0
        best_idx = -1

        for idx, gt in enumerate(class_gts):
            if idx in matched_gt_indices:
                continue
            iou = _compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt_indices.add(best_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    cum_tp = 0
    cum_fp = 0
    precisions: List[float] = []
    recalls:    List[float] = []
    n_gt = len(class_gts)

    for tp, fp in zip(tp_list, fp_list):
        cum_tp += tp
        cum_fp += fp
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / n_gt)

    ap = 0.0
    for recall_threshold in [t / 10 for t in range(11)]:
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= recall_threshold]
        ap += max(p_at_r) if p_at_r else 0.0

    return ap / 11.0


# Main evaluator
def evaluate(
    predictions:  List[List[Prediction]],
    ground_truth: List[List[BoundingBox]],
    latency:      float,
) -> ModelMetric:
    """
    Compute all detection metrics from per-frame prediction and GT lists.

    Parameters
    ----------
    predictions : List[List[Prediction]]
        One List[Prediction] per frame.
    ground_truth : List[List[BoundingBox]]
        One List[BoundingBox] per frame, same ordering as predictions.
    latency : float
        Seconds-per-frame measured externally by the caller.

    Returns
    -------
    ModelMetric
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions and ground_truth must have equal frame counts "
            f"(got {len(predictions)} vs {len(ground_truth)})"
        )

    all_preds = [p for frame in predictions for p in frame]
    all_gts   = [g for frame in ground_truth for g in frame]

    # accuracy and mean IoU 
    total_fn = 0
    total_tp = 0
    tp_ious: List[float] = []

    for preds, gts in zip(predictions, ground_truth):
        matched, unmatched_gt = _match_predictions(preds, gts, iou_threshold=0.5)
        total_tp += len(matched)
        total_fn += len(unmatched_gt)
        tp_ious.extend(iou for _, _, iou in matched)

    accuracy = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0 else 0.0
    )
    mean_iou = sum(tp_ious) / len(tp_ious) if tp_ious else 0.0

    # mAP_50 and mAP_95
    class_ids = list({g.class_id for g in all_gts})

    ap50_per_class = [
        _compute_ap_for_class(all_preds, all_gts, 0.50, cid)
        for cid in class_ids
    ]
    ap95_per_class = [
        _compute_ap_for_class(all_preds, all_gts, 0.95, cid)
        for cid in class_ids
    ]

    mAP_50 = sum(ap50_per_class) / len(ap50_per_class) if ap50_per_class else 0.0
    mAP_95 = sum(ap95_per_class) / len(ap95_per_class) if ap95_per_class else 0.0

    # weighted precision
    gt_count_per_class: Dict[int, int] = defaultdict(int)
    for gt in all_gts:
        gt_count_per_class[gt.class_id] += 1

    total_gt          = len(all_gts)
    weighted_precision = 0.0

    for class_id in class_ids:
        class_preds = [p for p in all_preds if p.class_id == class_id]
        class_gts   = [g for g in all_gts   if g.class_id == class_id]
        matched, _  = _match_predictions(class_preds, class_gts, iou_threshold=0.5)
        tp        = len(matched)
        fp        = len(class_preds) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        weight    = gt_count_per_class[class_id] / total_gt if total_gt > 0 else 0.0
        weighted_precision += precision * weight

    return ModelMetric(
        accuracy=accuracy,
        iou=mean_iou,
        mAP_50=mAP_50,
        mAP_95=mAP_95,
        weighted_precision=weighted_precision,
        latency=latency,
    )


# Visualisation
def visualize_predictions(
    predictions:     List[List[Prediction]],
    ground_truth:    List[List[BoundingBox]],
    frame_images:    List = None,
    max_frames:      int  = 4,
    save_path:       str  = None,
    class_names:     Dict[int, str] = None,
    max_preds_shown: int  = 10,
) -> None:
    """
    Draw GT (green) and predicted (red) bounding boxes for up to `max_frames` clips.

    Predictions are sorted by confidence and capped at `max_preds_shown` to keep
    the plot readable.  Any suppressed predictions are noted in the subplot title.
    """
    matplotlib.use("Agg" if save_path else "TkAgg")

    n_frames = len(predictions)
    if n_frames == 0:
        print("[eval] visualize_predictions: no frames to show.")
        return

    indices = [int(i) for i in np.linspace(0, n_frames - 1, min(max_frames, n_frames))]

    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 5))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        if frame_images is not None and idx < len(frame_images):
            img = np.array(frame_images[idx]).astype(np.float32)
            # Normalise float frames (e.g. residuals in [-1, 1]) to uint8
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            h, w = img.shape[:2]
            ax.imshow(img)
        else:
            h, w = 64, 64   # matches actual data resolution
            ax.imshow(np.full((h, w, 3), 200, dtype=np.uint8))

        def _draw_box(box, color, label):
            x0 = box.xmin * w
            y0 = box.ymin * h
            bw = (box.xmax - box.xmin) * w
            bh = (box.ymax - box.ymin) * h
            rect = patches.Rectangle(
                (x0, y0), bw, bh,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x0, max(y0 - 4, 0), label,
                color=color, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"),
            )

        # GT boxes — all valid annotations for this clip
        for gt in ground_truth[idx]:
            name = (class_names or {}).get(gt.class_id, f"cls {gt.class_id}")
            _draw_box(gt, color="lime", label=f"GT: {name}")

        # Predictions — top-K by confidence, rest suppressed
        clip_preds   = sorted(predictions[idx], key=lambda p: p.confidence, reverse=True)
        shown_preds  = clip_preds[:max_preds_shown]
        n_suppressed = len(clip_preds) - len(shown_preds)

        for pred in shown_preds:
            name     = (class_names or {}).get(pred.class_id, f"cls {pred.class_id}")
            iou_vals = [_compute_iou(pred, gt) for gt in ground_truth[idx]] if ground_truth[idx] else [0.0]
            best_iou = max(iou_vals) if iou_vals else 0.0
            _draw_box(pred, color="red", label=f"Pred: {name} {pred.confidence:.2f} IoU:{best_iou:.2f}")

        n_gt  = len(ground_truth[idx])
        title = f"Clip {idx}  |  GT={n_gt}  Pred={len(shown_preds)}"
        if n_suppressed:
            title += f"  (+{n_suppressed} suppressed)"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lime", linewidth=2, label="Ground Truth"),
        Line2D([0], [0], color="red",  linewidth=2, label="Prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Prediction Visualisation", fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[eval] Saved visualisation → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def visualize_temporal(
    clip:            dict,
    conf_threshold:  float = 0.5,
    max_preds_shown: int   = 10,
    save_path:       str   = None,
    class_names:     Dict[int, str] = None,
) -> None:
    """
    Unroll one clip across its T frames left-to-right to reveal temporal stability.

    Layout
    ------
    Row 1 (tall): T subplots — one per time-step.
      - Green box  : GT annotation present at that frame (empty if padding).
      - Red box    : top-K predictions (same query set shown every frame, since
                     the model outputs one set of Q queries per clip, not per frame).
      - Orange spine + "CLASS CHANGED" banner: the GT class at this frame differs
        from the GT class at the previous annotated frame — makes hopping obvious
        as you read left-to-right.
    Row 2 (thin): a colour-coded stability bar (steelblue = stable, orange = changed).

    Parameters
    ----------
    clip : dict  — one entry from temporal_clips built in validate():
        "images"   : List[np.ndarray(H,W,C)]      all T residual frames
        "gt_per_t" : List[List[BoundingBox]]       GT boxes per time-step (empty if no annotation)
        "boxes"    : np.ndarray [Q, 4]             clip-level predicted boxes
        "cls_ids"  : np.ndarray [Q]                argmax class per query
        "confs"    : np.ndarray [Q]                max softmax confidence per query
    """
    import matplotlib.gridspec as gridspec

    matplotlib.use("Agg" if save_path else "TkAgg")

    images   = clip["images"]    # List[ndarray]
    gt_per_t = clip["gt_per_t"]  # List[List[BoundingBox]]
    boxes    = clip["boxes"]     # [Q, 4]
    cls_ids  = clip["cls_ids"]   # [Q]
    confs    = clip["confs"]     # [Q]

    T = len(images)
    if T == 0:
        print("[eval] visualize_temporal: clip has no frames.")
        return

    # Top-K predictions sorted by confidence (same set shown at every frame)
    candidates = sorted(
        [(confs[q], q, int(cls_ids[q]))
         for q in range(len(confs)) if confs[q] >= conf_threshold],
        reverse=True,
    )[:max_preds_shown]

    top1_class = int(candidates[0][2]) if candidates else -1

    # Per-frame change flag: compare GT class at t vs previous annotated frame
    prev_gt_cls: int | None = None
    changed: List[bool] = []
    for t in range(T):
        if gt_per_t[t]:
            curr    = gt_per_t[t][0].class_id
            changed.append(prev_gt_cls is not None and curr != prev_gt_cls)
            prev_gt_cls = curr
        else:
            changed.append(False)   # no GT → can't determine change

    # Layout: tall image row + thin stability bar
    fig = plt.figure(figsize=(4 * T, 5))
    gs  = gridspec.GridSpec(2, T, height_ratios=[10, 1], hspace=0.35, wspace=0.08)

    for t in range(T):
        ax = fig.add_subplot(gs[0, t])

        # Normalise residual frame to uint8 for display
        img = images[t].astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        h, w = img.shape[:2]
        ax.imshow(img)

        def _draw_box(box, color, label):
            x0 = box.xmin * w
            y0 = box.ymin * h
            bw = (box.xmax - box.xmin) * w
            bh = (box.ymax - box.ymin) * h
            rect = patches.Rectangle(
                (x0, y0), bw, bh,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x0, max(y0 - 4, 0), label,
                color=color, fontsize=7,
                bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"),
            )

        # GT for this time-step
        for gt in gt_per_t[t]:
            name = (class_names or {}).get(gt.class_id, f"cls {gt.class_id}")
            _draw_box(gt, color="lime", label=f"GT:{name}")

        # Top-K predictions (same every frame — clip-level outputs)
        for conf, q, cls_id in candidates:
            class _B: pass
            b = _B()
            b.xmin, b.xmax, b.ymin, b.ymax = (
                float(boxes[q, 0]), float(boxes[q, 1]),
                float(boxes[q, 2]), float(boxes[q, 3]),
            )
            name     = (class_names or {}).get(cls_id, f"cls {cls_id}")
            iou_vals = [_compute_iou(b, gt) for gt in gt_per_t[t]] if gt_per_t[t] else [0.0]
            best_iou = max(iou_vals) if iou_vals else 0.0
            _draw_box(b, color="red", label=f"{name} {conf:.2f} IoU:{best_iou:.2f}")

        # Subplot title
        gt_cls_str = f"cls {gt_per_t[t][0].class_id}" if gt_per_t[t] else "no GT"
        ax.set_title(f"t={t}  {gt_cls_str}", fontsize=8)
        ax.axis("off")

        # Orange spine + "CLASS CHANGED" banner when GT class changes
        if changed[t]:
            for spine in ax.spines.values():
                spine.set_edgecolor("orange")
                spine.set_linewidth(3)
                spine.set_visible(True)
            ax.text(
                0.5, 0.97, "CLASS CHANGED",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=7, fontweight="bold", color="orange",
                bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"),
            )

    # Stability bar — one coloured cell per time-step
    ax_bar = fig.add_subplot(gs[1, :])
    for t in range(T):
        color = "orange" if changed[t] else "steelblue"
        ax_bar.barh(0, 1, left=t, color=color, edgecolor="white", linewidth=0.5)
        ax_bar.text(
            t + 0.5, 0, "change" if changed[t] else "stable",
            ha="center", va="center", fontsize=7, color="white", fontweight="bold",
        )
    ax_bar.set_xlim(0, T)
    ax_bar.set_yticks([])
    ax_bar.set_xticks(range(T))
    ax_bar.set_xticklabels([f"t={t}" for t in range(T)], fontsize=7)
    ax_bar.set_title("GT class stability across clip", fontsize=8, pad=2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lime",      linewidth=2, label="Ground Truth"),
        Line2D([0], [0], color="red",       linewidth=2, label="Prediction"),
        Line2D([0], [0], color="orange",    linewidth=3, label="GT class changed"),
        Line2D([0], [0], color="steelblue", linewidth=3, label="GT class stable"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Temporal stability  |  top-1 pred = cls {top1_class}  |  T={T} frames",
        fontsize=11, y=1.01,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[eval] Saved temporal visualisation → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# Self-test  (python eval_framework.py)

if __name__ == "__main__":

    def _box(xmin, xmax, ymin, ymax, cls):
        return BoundingBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=cls)

    def _pred(xmin, xmax, ymin, ymax, cls, conf):
        return Prediction(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=cls, confidence=conf)

    print("=" * 60)
    print("Case 1: Perfect predictions")
    preds_1 = [[_pred(0.1, 0.6, 0.1, 0.6, cls=0, conf=0.95)]]
    gts_1   = [[_box(0.1, 0.6, 0.1, 0.6, cls=0)]]
    m1 = evaluate(preds_1, gts_1, latency=0.012)
    print(m1)
    assert m1.accuracy == 1.0,           f"Expected 1.0, got {m1.accuracy}"
    assert abs(m1.iou - 1.0) < 1e-5,    f"Expected IoU~1.0, got {m1.iou}"
    assert m1.mAP_50 == 1.0,            f"Expected mAP_50=1.0, got {m1.mAP_50}"
    print("  PASS\n")

    print("=" * 60)
    print("Case 2: Partial overlap (IoU < 0.5 → no match)")
    # GT spans [0.0, 0.7] in x, pred spans [0.3, 1.0] → overlap = [0.3,0.7]=0.4, union=1.0 → IoU=0.4
    preds_2 = [[_pred(0.3, 1.0, 0.0, 1.0, cls=0, conf=0.8)]]
    gts_2   = [[_box(0.0, 0.7, 0.0, 1.0, cls=0)]]
    m2 = evaluate(preds_2, gts_2, latency=0.015)
    print(m2)
    assert m2.accuracy == 0.0, f"Expected 0.0, got {m2.accuracy}"
    assert m2.iou      == 0.0, f"Expected 0.0, got {m2.iou}"
    print("  PASS\n")

    print("=" * 60)
    print("Case 3: No overlap")
    preds_3 = [[_pred(0.0, 0.3, 0.0, 0.3, cls=0, conf=0.9)]]
    gts_3   = [[_box(0.7, 1.0, 0.7, 1.0, cls=0)]]
    m3 = evaluate(preds_3, gts_3, latency=0.011)
    print(m3)
    assert m3.accuracy == 0.0, f"Expected 0.0, got {m3.accuracy}"
    assert m3.iou      == 0.0, f"Expected 0.0, got {m3.iou}"
    assert m3.mAP_50   == 0.0, f"Expected 0.0, got {m3.mAP_50}"
    print("  PASS\n")

    print("=" * 60)
    print("Case 4: Multi-class, two frames")
    preds_4 = [
        [
            _pred(0.1, 0.5, 0.1, 0.5, cls=0, conf=0.9),
            _pred(0.6, 0.9, 0.6, 0.9, cls=1, conf=0.8),
        ],
        [_pred(0.2, 0.6, 0.2, 0.6, cls=0, conf=0.7)],
    ]
    gts_4 = [
        [
            _box(0.1, 0.5, 0.1, 0.5, cls=0),
            _box(0.6, 0.9, 0.6, 0.9, cls=1),
        ],
        [_box(0.2, 0.6, 0.2, 0.6, cls=0)],
    ]
    m4 = evaluate(preds_4, gts_4, latency=0.013)
    print(m4)
    assert m4.accuracy == 1.0,        f"Expected 1.0, got {m4.accuracy}"
    assert abs(m4.iou - 1.0) < 1e-5, f"Expected IoU~1.0, got {m4.iou}"
    print("  PASS\n")

    print("=" * 60)
    print("compare(): m1 (perfect) vs m3 (no overlap)")
    diff = m1.compare(m3)
    for key, value in diff.items():
        print(f"  {key}: {value:+.2f}%")

    print("\nAll checks passed.")