"""
evaluation.py - Evaluation framework for compressed-domain object detection.

Usage:
    from evaluation import BoundingBox, Prediction, ModelMetric, evaluate
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    x: float       # top-left x, normalized 0-1
    y: float       # top-left y, normalized 0-1
    w: float       # width, normalized 0-1
    h: float       # height, normalized 0-1
    class_id: int


@dataclass
class Prediction(BoundingBox):
    confidence: float  # 0-1, higher = more confident


@dataclass
class ModelMetric:
    accuracy: float            # TP / (TP + FN) across all frames
    iou: float                 # mean IoU over matched TP pairs only
    mAP_50: float              # macro-averaged AP per class at IoU 0.50
    mAP_95: float              # single threshold AP at IoU 0.95
    weighted_precision: float  # per-class precision at IoU 0.5, weighted by GT count
    latency: float             # seconds per frame, passed in from caller

    def compare(self, other: "ModelMetric") -> Dict[str, float]:
        """
        Return percentage difference between self and other for each metric.

        Positive values mean self is higher.
        Negative values mean self is lower.
        Used in benchmark chapter to compare YOLO baseline vs ConvLSTM model.
        """
        def _pct(a: float, b: float) -> float:
            if b == 0.0:
                return float("inf") if a != 0.0 else 0.0
            return (a - b) / b * 100.0

        return {
            "accuracy_diff_pct": _pct(self.accuracy, other.accuracy),
            "iou_diff_pct": _pct(self.iou, other.iou),
            "mAP_50_diff_pct": _pct(self.mAP_50, other.mAP_50),
            "mAP_95_diff_pct": _pct(self.mAP_95, other.mAP_95),
            "precision_diff_pct": _pct(self.weighted_precision, other.weighted_precision),
            "latency_diff_pct": _pct(self.latency, other.latency),
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


# ---------------------------------------------------------------------------
# IOU helper
# ---------------------------------------------------------------------------

def _compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute IoU between two boxes in xywh normalized format."""
    ax1 = box_a.x
    ay1 = box_a.y
    ax2 = box_a.x + box_a.w
    ay2 = box_a.y + box_a.h

    bx1 = box_b.x
    by1 = box_b.y
    bx2 = box_b.x + box_b.w
    by2 = box_b.y + box_b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = box_a.w * box_a.h
    area_b = box_b.w * box_b.h
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Matching helper
# ---------------------------------------------------------------------------

def _match_predictions(
    predictions: List[Prediction],
    ground_truth: List[BoundingBox],
    iou_threshold: float,
) -> Tuple[List[Tuple[Prediction, BoundingBox, float]], List[BoundingBox]]:
    """
    Greedily match predictions to ground truth boxes at a given IoU threshold.

    Predictions are processed in descending confidence order.
    Each ground truth box can only be matched once.

    Returns matched pairs with their IoU scores, and unmatched ground truth boxes.
    """
    matched: List[Tuple[Prediction, BoundingBox, float]] = []
    unmatched_gt = list(ground_truth)

    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt = None

        for gt in unmatched_gt:
            if gt.class_id != pred.class_id:
                continue
            iou = _compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt is not None and best_iou >= iou_threshold:
            matched.append((pred, best_gt, best_iou))
            unmatched_gt.remove(best_gt)

    return matched, unmatched_gt


# ---------------------------------------------------------------------------
# AP helper
# ---------------------------------------------------------------------------

def _compute_ap_for_class(
    predictions: List[Prediction],
    ground_truth: List[BoundingBox],
    iou_threshold: float,
    class_id: int,
) -> float:
    """
    Compute Average Precision for a single class at a given IoU threshold.

    Uses 11-point interpolation over the precision-recall curve.
    Returns 0.0 if there are no ground truth boxes for this class.
    """
    class_preds = [p for p in predictions if p.class_id == class_id]
    class_gts = [g for g in ground_truth if g.class_id == class_id]

    if not class_gts or not class_preds:
        return 0.0

    class_preds = sorted(class_preds, key=lambda p: p.confidence, reverse=True)

    matched_gt_indices = set()
    tp_list = []
    fp_list = []

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

    cumulative_tp = 0
    cumulative_fp = 0
    precisions = []
    recalls = []
    n_gt = len(class_gts)

    for tp, fp in zip(tp_list, fp_list):
        cumulative_tp += tp
        cumulative_fp += fp
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
        recalls.append(cumulative_tp / n_gt)

    ap = 0.0
    for recall_threshold in [t / 10 for t in range(11)]:
        precision_at_recall = [
            p for p, r in zip(precisions, recalls)
            if r >= recall_threshold
        ]
        ap += max(precision_at_recall) if precision_at_recall else 0.0

    return ap / 11


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate(
    predictions: List[List[Prediction]],
    ground_truth: List[List[BoundingBox]],
    latency: float,
) -> ModelMetric:
    """
    Compute all metrics from predictions and ground truth boxes.

    Parameters
    ----------
    predictions:
        One List[Prediction] per frame.
    ground_truth:
        One List[BoundingBox] per frame, matching predictions index.
    latency:
        Seconds per frame, measured externally by the caller.

    Returns
    -------
    ModelMetric with all fields populated.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions and ground_truth must have the same number of frames "
            f"(got {len(predictions)} vs {len(ground_truth)})"
        )

    all_preds = [p for frame in predictions for p in frame]
    all_gts = [g for frame in ground_truth for g in frame]

    # --- accuracy and mean IoU ---------------------------------------------

    total_tp = 0
    total_fn = 0
    tp_ious: List[float] = []

    for preds, gts in zip(predictions, ground_truth):
        matched, unmatched_gt = _match_predictions(preds, gts, iou_threshold=0.5)
        total_tp += len(matched)
        total_fn += len(unmatched_gt)
        tp_ious.extend(iou for _, _, iou in matched)

    accuracy = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )
    mean_iou = sum(tp_ious) / len(tp_ious) if tp_ious else 0.0

    # --- mAP_50 and mAP_95 -------------------------------------------------

    class_ids = list({g.class_id for g in all_gts})

    ap50_per_class = [
        _compute_ap_for_class(all_preds, all_gts, 0.50, class_id)
        for class_id in class_ids
    ]
    ap95_per_class = [
        _compute_ap_for_class(all_preds, all_gts, 0.95, class_id)
        for class_id in class_ids
    ]

    mAP_50 = sum(ap50_per_class) / len(ap50_per_class) if ap50_per_class else 0.0
    mAP_95 = sum(ap95_per_class) / len(ap95_per_class) if ap95_per_class else 0.0

    # --- weighted precision -------------------------------------------------

    gt_count_per_class: Dict[int, int] = defaultdict(int)
    for gt in all_gts:
        gt_count_per_class[gt.class_id] += 1

    total_gt = len(all_gts)
    weighted_precision = 0.0

    for class_id in class_ids:
        class_preds = [p for p in all_preds if p.class_id == class_id]
        class_gts = [g for g in all_gts if g.class_id == class_id]

        matched, _ = _match_predictions(class_preds, class_gts, iou_threshold=0.5)
        tp = len(matched)
        fp = len(class_preds) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        weight = gt_count_per_class[class_id] / total_gt if total_gt > 0 else 0.0
        weighted_precision += precision * weight

    return ModelMetric(
        accuracy=accuracy,
        iou=mean_iou,
        mAP_50=mAP_50,
        mAP_95=mAP_95,
        weighted_precision=weighted_precision,
        latency=latency,
    )


# ---------------------------------------------------------------------------
# Placeholder tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _box(x, y, w, h, cls):
        return BoundingBox(x=x, y=y, w=w, h=h, class_id=cls)

    def _pred(x, y, w, h, cls, conf):
        return Prediction(x=x, y=y, w=w, h=h, class_id=cls, confidence=conf)

    print("=" * 60)
    print("Case 1: Perfect predictions")
    preds_1 = [[_pred(0.1, 0.1, 0.5, 0.5, cls=0, conf=0.95)]]
    gts_1 = [[_box(0.1, 0.1, 0.5, 0.5, cls=0)]]
    m1 = evaluate(preds_1, gts_1, latency=0.012)
    print(m1)
    assert m1.accuracy == 1.0
    assert abs(m1.iou - 1.0) < 1e-5
    assert m1.mAP_50 == 1.0
    print("  PASS\n")

    print("=" * 60)
    print("Case 2: Partial overlap")
    preds_2 = [[_pred(0.3, 0.0, 0.7, 1.0, cls=0, conf=0.8)]]
    gts_2 = [[_box(0.0, 0.0, 0.7, 1.0, cls=0)]]
    m2 = evaluate(preds_2, gts_2, latency=0.015)
    print(m2)
    assert m2.accuracy == 0.0
    assert m2.iou == 0.0
    print("  PASS\n")

    print("=" * 60)
    print("Case 3: No overlap")
    preds_3 = [[_pred(0.0, 0.0, 0.3, 0.3, cls=0, conf=0.9)]]
    gts_3 = [[_box(0.7, 0.7, 0.3, 0.3, cls=0)]]
    m3 = evaluate(preds_3, gts_3, latency=0.011)
    print(m3)
    assert m3.accuracy == 0.0
    assert m3.iou == 0.0
    assert m3.mAP_50 == 0.0
    print("  PASS\n")

    print("=" * 60)
    print("Case 4: Multi-class, two frames")
    preds_4 = [
        [
            _pred(0.1, 0.1, 0.4, 0.4, cls=0, conf=0.9),
            _pred(0.6, 0.6, 0.3, 0.3, cls=1, conf=0.8),
        ],
        [_pred(0.2, 0.2, 0.4, 0.4, cls=0, conf=0.7)],
    ]
    gts_4 = [
        [
            _box(0.1, 0.1, 0.4, 0.4, cls=0),
            _box(0.6, 0.6, 0.3, 0.3, cls=1),
        ],
        [_box(0.2, 0.2, 0.4, 0.4, cls=0)],
    ]
    m4 = evaluate(preds_4, gts_4, latency=0.013)
    print(m4)
    assert m4.accuracy == 1.0
    assert abs(m4.iou - 1.0) < 1e-5
    print("  PASS\n")

    print("=" * 60)
    print("compare(): m1 (perfect) vs m3 (no overlap)")
    diff = m1.compare(m3)
    for key, value in diff.items():
        print(f"  {key}: {value:+.2f}%")

    print("\nAll checks passed.")