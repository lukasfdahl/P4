"""
faster_rcnn.py - Faster R-CNN baseline for comparison against ObjectDetector.

Wraps torchvisions Faster R-CNN (ResNet-50 + FPN) so it fits the same
train/eval loop as ObjectDetector. The forward signature is identical so
the same loss function and eval framework work without any changes.

Inputs are the mean of residuals and motion vectors across the clip,
collapsing the temporal dimension into a single frame. MVs are upsampled
to frame resolution and projected together with residuals to 3 channels
before being fed to the backbone.

The detector always runs in inference mode even during training, because
torchvisions training path uses its own internal losses which are not
compatible with the Hungarian matching loss used here. Gradients still
flow through the backbone since torch.set_grad_enabled mirrors self.training,
so the weights update normally.
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

    def __init__(
        self,
        num_classes: int   = 23,
        num_queries: int   = 10,
        pretrained:  bool  = False,
        min_score:   float = 0.05,
    ):
        super().__init__()

        self.num_queries        = num_queries
        self.num_object_classes = num_classes
        self.no_object_class    = num_classes   # background slot, same convention as ObjectDetector
        self.num_output_classes = num_classes + 1
        self.min_score          = min_score

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.detector = fasterrcnn_resnet50_fpn(
            weights  = weights,
            min_size = 64,   # dont let torchvision secretly upscale our 64x64 frames
            max_size = 64,
        )

        # swap out the classification head to match our number of classes
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes + 1,  # +1 for torchvisions background class 0
        )

        # cap detections per image so output shape is always [B, Q, *]
        self.detector.roi_heads.detections_per_img = num_queries

        # project residuals (3ch) + upsampled MVs (4ch) -> 3ch for the backbone
        self.mv_proj = nn.Conv2d(7, 3, kernel_size=1)

    def forward(
        self,
        motion_vectors: torch.Tensor | None,  # [B, T, 4, H_mv, W_mv]
        residuals:      torch.Tensor,          # [B, T, 3, H, W]
        iframe_mask:    torch.Tensor,          # [B, T] bool - not used
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, C, H, W = residuals.shape
        device = residuals.device

        # average both inputs across the clip to get a single frame per sample
        res_mean = residuals.mean(dim=1).clamp(0.0, 1.0)   # [B, 3, H, W]

        if motion_vectors is not None:
            # upsample MVs from macroblock grid to frame resolution, then concat with residuals
            mv_mean = motion_vectors.float().mean(dim=1)                          # [B, 4, H_mv, W_mv]
            mv_up   = F.interpolate(mv_mean, size=(H, W), mode="bilinear",
                                    align_corners=False)                          # [B, 4, H, W]
            x = self.mv_proj(torch.cat([res_mean, mv_up], dim=1))                # [B, 3, H, W]
        else:
            x = res_mean

        x = x.clamp(0.0, 1.0)

        # run in inference mode - torchvisions training path is incompatible with
        # our hungarian loss, so we grab the raw detections and re-score them ourselves.
        # gradients still flow because set_grad_enabled mirrors self.training.
        was_training = self.training
        self.detector.eval()

        with torch.set_grad_enabled(was_training):
            image_list  = [x[b] for b in range(B)]
            raw_outputs = self.detector(image_list)   # list of dicts, one per image

        if was_training:
            self.detector.train()

        all_boxes   = []
        all_classes = []

        for b in range(B):
            det        = raw_outputs[b]
            det_boxes  = det["boxes"]    # [N, 4] pixel coords [xmin, ymin, xmax, ymax]
            det_scores = det["scores"]   # [N]
            det_labels = det["labels"]   # [N] 1-indexed (torchvision convention)
            N = det_boxes.shape[0]

            # normalise to [0,1] and convert to [xmin, xmax, ymin, ymax]
            if N > 0:
                xmin = det_boxes[:, 0] / W
                xmax = det_boxes[:, 2] / W
                ymin = det_boxes[:, 1] / H
                ymax = det_boxes[:, 3] / H
                norm_boxes = torch.stack([xmin, xmax, ymin, ymax], dim=1)  # [N, 4]
            else:
                norm_boxes = det_boxes.new_zeros(0, 4)

            # build logits from detection scores so the hungarian matcher gets proper signal
            if N > 0:
                eps         = 1e-6
                obj_logit   = torch.log(det_scores + eps) - torch.log(1 - det_scores + eps)
                noobj_logit = -obj_logit

                logits  = det_boxes.new_full((N, self.num_output_classes), -10.0)
                cls_idx = (det_labels - 1).clamp(0, self.num_object_classes - 1)
                logits[torch.arange(N), cls_idx]              = obj_logit
                logits[torch.arange(N), self.no_object_class] = noobj_logit
            else:
                logits = det_boxes.new_full((0, self.num_output_classes), -10.0)

            # pad or truncate to exactly num_queries slots
            Q = self.num_queries
            if N >= Q:
                boxes_out  = norm_boxes[:Q]
                logits_out = logits[:Q]
            else:
                pad = Q - N
                pad_boxes  = norm_boxes.new_full((pad, 4), 0.5)
                pad_boxes[:, 0] = 0.25; pad_boxes[:, 1] = 0.75   # xmin/xmax
                pad_boxes[:, 2] = 0.25; pad_boxes[:, 3] = 0.75   # ymin/ymax
                pad_logits = logits.new_full((pad, self.num_output_classes), -10.0)
                pad_logits[:, self.no_object_class] = 10.0        # strongly predict no-object

                boxes_out  = torch.cat([norm_boxes, pad_boxes],  dim=0)  # [Q, 4]
                logits_out = torch.cat([logits,     pad_logits], dim=0)  # [Q, C+1]

            all_boxes.append(boxes_out)
            all_classes.append(logits_out)

        return torch.stack(all_boxes), torch.stack(all_classes)  # [B,Q,4], [B,Q,C+1]


# smoke test
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