"""
faster_rcnn.py  –  Faster R-CNN baseline for comparison against ObjectDetector.

Uses torchvision's complete Faster R-CNN implementation (ResNet-50 + FPN backbone,
RPN, RoI pooling) with minimal wrapping so it fits the same train/eval loop.

Key design decisions
--------------------
- Forward signature matches ObjectDetector exactly: (motion_vectors, residuals,
  iframe_mask) → (boxes [B,Q,4], classes [B,Q,C+1])
  This means the SAME train loop, loss function, and eval framework work unchanged.
- Input: mean of residuals across the clip (T frames → 1 RGB image per sample).
  Faster R-CNN is a single-frame model; averaging the clip is the fairest way to
  give it the same temporal window your model sees.
- Box format: your codebase uses [xmin, xmax, ymin, ymax] normalised to [0,1].
  Faster R-CNN internally uses [xmin, ymin, xmax, ymax] in pixel coords —
  this wrapper converts in both directions transparently.
- Output: fixed Q=num_queries slots (same as ObjectDetector) filled from the
  top-scoring detections, padded with no-object if fewer than Q are returned.
  This means the Hungarian-matching loss works without modification.

Why Faster R-CNN is the right comparison
-----------------------------------------
- It is the canonical two-stage detector — well understood, widely cited.
- It operates on decoded RGB pixels, making it the "standard pipeline" baseline
  that your compressed-domain model aims to replace or match.
- pretrained=True gives the ImageNet+COCO ceiling; pretrained=False gives a
  from-scratch comparison on equal footing with your model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNDetector(nn.Module):
    """
    Faster R-CNN (ResNet-50 + FPN) wrapped to match ObjectDetector's interface.

    Forward signature:
        forward(motion_vectors, residuals, iframe_mask) -> (boxes, classes)

    Parameters
    ----------
    num_classes : int
        Number of object classes (no-object added automatically, same as ObjectDetector).
    num_queries : int
        Number of output detection slots. Top-N detections fill slots; remainder
        are padded as no-object. Use the same value as ObjectDetector (default 10).
    pretrained : bool
        True  → COCO-pretrained weights (best RGB ceiling).
        False → random init, fairest from-scratch comparison.
    min_score : float
        Confidence threshold below which detections are treated as background.
        0.05 is intentionally low — the loss handles ranking, not this threshold.
    """

    def __init__(
        self,
        num_classes: int  = 23,
        num_queries: int  = 10,
        pretrained:  bool = False,
        min_score:   float = 0.05,
    ):
        super().__init__()

        self.num_queries        = num_queries
        self.num_object_classes = num_classes
        self.no_object_class    = num_classes      # background slot index (same convention)
        self.num_output_classes = num_classes + 1
        self.min_score          = min_score

        # ── Build Faster R-CNN ────────────────────────────────────────────────
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.detector = fasterrcnn_resnet50_fpn(
            weights          = weights,
            # Disable the internal min_size resize — our frames are already 64x64
            # and we don't want torchvision secretly upscaling them.
            min_size         = 64,
            max_size         = 64,
        )

        # Replace the classification head to match our num_classes + background.
        # torchvision's convention: class 0 = background, 1..N = objects.
        # We use num_classes+1 outputs to match ObjectDetector's no-object slot.
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes + 1,  # +1 for torchvision's background class 0
        )

        # Cap the number of detections returned to num_queries so output size
        # is fixed and predictable — same as ObjectDetector's num_queries slots.
        self.detector.roi_heads.detections_per_img = num_queries

    def forward(
        self,
        motion_vectors: torch.Tensor | None,  # [B, T, 4, H_mv, W_mv] — not used
        residuals:      torch.Tensor,          # [B, T, 3, H, W]
        iframe_mask:    torch.Tensor,          # [B, T] bool — not used
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        boxes   : [B, Q, 4]  normalised [xmin, xmax, ymin, ymax]  (same as ObjectDetector)
        classes : [B, Q, C+1] logits                               (same as ObjectDetector)
        """
        B, T, C, H, W = residuals.shape
        device = residuals.device

        # Temporal fusion: mean across clip frames → one image per sample
        x = residuals.mean(dim=1).clamp(0.0, 1.0)   # [B, 3, H, W]

        # ── Run Faster R-CNN in eval mode (inference path) ────────────────────
        # We always use the inference path because Faster R-CNN's training path
        # requires targets in a different format and computes its own internal
        # losses. We take the raw detections and re-score them with our own
        # Hungarian-matching loss, keeping everything consistent.
        was_training = self.training
        self.detector.eval()

        with torch.set_grad_enabled(was_training):
            # torchvision expects a list of [3, H, W] tensors
            image_list   = [x[b] for b in range(B)]
            raw_outputs  = self.detector(image_list)  # list of dicts, one per image

        if was_training:
            self.detector.train()

        # ── Convert detections to fixed-size [B, Q, *] tensors ───────────────
        all_boxes   = []
        all_classes = []

        for b in range(B):
            det      = raw_outputs[b]
            det_boxes  = det["boxes"]    # [N, 4] pixel [xmin, ymin, xmax, ymax]
            det_scores = det["scores"]   # [N]
            det_labels = det["labels"]   # [N]  1-indexed (0 = background in torchvision)
            N = det_boxes.shape[0]

            # ── Normalise boxes from pixel to [0,1] ───────────────────────────
            # and convert [xmin, ymin, xmax, ymax] → [xmin, xmax, ymin, ymax]
            if N > 0:
                xmin = det_boxes[:, 0] / W
                xmax = det_boxes[:, 2] / W
                ymin = det_boxes[:, 1] / H
                ymax = det_boxes[:, 3] / H
                norm_boxes = torch.stack([xmin, xmax, ymin, ymax], dim=1)  # [N, 4]
            else:
                norm_boxes = det_boxes.new_zeros(0, 4)

            # ── Build logit tensor [N, C+1] ───────────────────────────────────
            # We construct soft logits from the detection scores:
            #   - The detected class slot gets log(score / (1 - score + eps))
            #   - The no-object slot gets log((1-score) / (score + eps))
            # This is a reasonable approximation that gives the Hungarian matcher
            # proper signal without re-running a separate classification head.
            if N > 0:
                eps        = 1e-6
                obj_logit  = torch.log(det_scores + eps) - torch.log(1 - det_scores + eps)
                noobj_logit = -obj_logit

                logits = det_boxes.new_full((N, self.num_output_classes), -10.0)
                # Map torchvision labels (1-indexed) to our 0-indexed class IDs
                cls_idx = (det_labels - 1).clamp(0, self.num_object_classes - 1)
                logits[torch.arange(N), cls_idx]           = obj_logit
                logits[torch.arange(N), self.no_object_class] = noobj_logit
            else:
                logits = det_boxes.new_full((0, self.num_output_classes), -10.0)

            # ── Pad / truncate to exactly num_queries slots ───────────────────
            Q = self.num_queries
            if N >= Q:
                # Already capped at num_queries by roi_heads.detections_per_img,
                # but handle edge case where N > Q anyway.
                boxes_out  = norm_boxes[:Q]
                logits_out = logits[:Q]
            else:
                # Pad remaining slots as no-object centred boxes
                pad = Q - N
                pad_boxes  = norm_boxes.new_full((pad, 4), 0.5)   # centred unit box
                pad_boxes[:, 0] = 0.25; pad_boxes[:, 1] = 0.75   # xmin/xmax
                pad_boxes[:, 2] = 0.25; pad_boxes[:, 3] = 0.75   # ymin/ymax
                pad_logits = logits.new_full((pad, self.num_output_classes), -10.0)
                pad_logits[:, self.no_object_class] = 10.0        # strong no-object

                boxes_out  = torch.cat([norm_boxes, pad_boxes],  dim=0)  # [Q, 4]
                logits_out = torch.cat([logits,     pad_logits], dim=0)  # [Q, C+1]

            all_boxes.append(boxes_out)
            all_classes.append(logits_out)

        return torch.stack(all_boxes), torch.stack(all_classes)  # [B,Q,4], [B,Q,C+1]


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BATCH, T, H, W = 2, 5, 64, 64
    BASE_MV        = 16
    NUM_CLASSES    = 23
    NUM_QUERIES    = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv     = torch.randn(BATCH, T, 4, H // BASE_MV, W // BASE_MV).to(device)
    res    = torch.rand (BATCH, T, 3, H, W).to(device)
    imask  = torch.zeros(BATCH, T, dtype=torch.bool).to(device)
    imask[:, 0] = True

    for pretrained in (False, True):
        model = FasterRCNNDetector(
            num_classes=NUM_CLASSES, num_queries=NUM_QUERIES,
            pretrained=pretrained,
        ).to(device)
        model.eval()

        boxes, classes = model(mv, res, imask)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[pretrained={str(pretrained):5s}]  "
              f"boxes {tuple(boxes.shape)}  classes {tuple(classes.shape)}  "
              f"params: {n:,}")