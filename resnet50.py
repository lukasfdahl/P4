"""
benchmark_model.py  –  ResNet-50 baseline for comparison against ObjectDetector.

Imports ResNet-50 directly from torchvision (no architecture redefinition) and
reuses PredictionHeads from model.py.  The only change to the stock ResNet is
replacing conv1 to accept 7 input channels (3 residual + 4 MV) instead of 3.

Input modes
-----------
  "residuals_only"  (3 ch)  — pixels only.
  "mv_concat"       (7 ch)  — residuals + motion vectors concatenated.

Pretrained flag
---------------
  pretrained=False  — train from scratch, fairest architectural comparison.
  pretrained=True   — ImageNet init, useful as a "best RGB baseline" ceiling.

Drop-in usage in train.py
--------------------------
  # from model import ObjectDetector
  # model = ObjectDetector(...)
  from benchmark_model import ResNet50Detector
  model = ResNet50Detector(num_classes=..., num_queries=...,
                           input_mode="mv_concat", pretrained=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# Reuse heads directly from your model — no duplication
from model import PredictionHeads


class ResNet50Detector(nn.Module):
    """
    ResNet-50 backbone → global average pool → num_queries object queries.

    Forward signature mirrors ObjectDetector exactly:
        forward(motion_vectors, residuals, iframe_mask) -> (boxes, classes)

    Parameters
    ----------
    num_classes : int
        Number of object classes (no-object class added automatically).
    num_queries : int
        Number of detection slots — use the same value as ObjectDetector.
    hidden_dim : int
        Width of the query projection and prediction heads.
    input_mode : str
        "residuals_only"  — 3-channel input, MVs ignored.
        "mv_concat"       — 7-channel input (3 residual + 4 MV).
    pretrained : bool
        Load ImageNet weights. Set False for a fair from-scratch comparison.
    base_mv_scale : int
        Native H.264 MV block size (default 16). Used to upsample MVs to
        full frame resolution before concatenation.
    freeze_backbone : bool
        Freeze all ResNet-50 parameters after construction.
    """

    _VALID_MODES = {"residuals_only", "mv_concat"}

    def __init__(
        self,
        num_classes:     int  = 10,
        num_queries:     int  = 10,
        hidden_dim:      int  = 256,
        input_mode:      str  = "mv_concat",
        pretrained:      bool = False,
        base_mv_scale:   int  = 16,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if input_mode not in self._VALID_MODES:
            raise ValueError(f"input_mode must be one of {self._VALID_MODES}, got '{input_mode}'")

        self.input_mode         = input_mode
        self.base_mv_scale      = base_mv_scale
        self.num_queries        = num_queries
        self.num_object_classes = num_classes
        self.no_object_class    = num_classes       # same convention as ObjectDetector
        self.num_output_classes = num_classes + 1
        self.hidden_dim         = hidden_dim

        # ── ResNet-50 from torchvision, unmodified except conv1 ───────────────
        weights  = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        if input_mode == "mv_concat":
            # Only change to the stock ResNet: swap conv1 for a 7-channel version.
            # RGB slice gets the pretrained weights; MV slice is zero-initialised.
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels  = 7,
                out_channels = old_conv.out_channels,
                kernel_size  = old_conv.kernel_size,
                stride       = old_conv.stride,
                padding      = old_conv.padding,
                bias         = old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight   # RGB — pretrained or random
                new_conv.weight[:, 3:, :, :] = 0.0               # MV  — learn from scratch
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            backbone.conv1 = new_conv

        # Strip the 1000-class FC; keep everything up to avgpool → [B, 2048, 1, 1]
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ── Query projection ──────────────────────────────────────────────────
        self.query_proj = nn.Linear(2048, num_queries * hidden_dim)

        # ── Prediction heads — imported directly from model.py ────────────────
        self.prediction_heads = PredictionHeads(
            hidden_dim       = hidden_dim,
            num_classes      = self.num_output_classes,
            bbox_head_dim    = hidden_dim,
            class_head_dim   = hidden_dim,
            bbox_num_layers  = 3,
            class_num_layers = 3,
        )

    def forward(
        self,
        motion_vectors: torch.Tensor,   # [B, T, 4, H_mv, W_mv]
        residuals:      torch.Tensor,   # [B, T, 3, H, W]
        iframe_mask:    torch.Tensor,   # [B, T] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, _, H, W = residuals.shape

        # Zero I-frame MVs — mirrors ObjectDetector's convention
        mv_flat    = motion_vectors.reshape(B * T, *motion_vectors.shape[2:]).clone()
        imask_flat = iframe_mask.to(device=mv_flat.device, dtype=torch.bool).reshape(B * T)
        mv_flat[imask_flat] = 0.0
        mv = mv_flat.view(B, T, *motion_vectors.shape[2:])

        # Temporal fusion: mean across clip
        res_mean = residuals.mean(dim=1)    # [B, 3, H, W]
        mv_mean  = mv.mean(dim=1)           # [B, 4, H_mv, W_mv]

        res_clamped = res_mean.clamp(0.0, 1.0)

        if self.input_mode == "residuals_only":
            x = res_clamped                 # [B, 3, H, W]
        else:
            # Upsample MVs from coarse MV grid to full frame resolution
            mv_up = F.interpolate(mv_mean.float(), size=(H, W),
                                  mode="bilinear", align_corners=False)
            x = torch.cat([res_clamped, mv_up], dim=1)   # [B, 7, H, W]

        # Backbone → [B, 2048] (avgpool + flattened by the Identity fc)
        feat    = self.backbone(x).flatten(1)

        # Project to Q query vectors
        queries = self.query_proj(feat).view(B, self.num_queries, self.hidden_dim)

        return self.prediction_heads(queries)   # (boxes [B,Q,4], classes [B,Q,C+1])


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BATCH, T, H, W  = 4, 5, 64, 64
    BASE_MV         = 16
    NUM_CLASSES     = 10
    NUM_QUERIES     = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv     = torch.randn(BATCH, T, 4, H // BASE_MV, W // BASE_MV).to(device)
    res    = torch.rand (BATCH, T, 3, H, W).to(device)
    imask  = torch.zeros(BATCH, T, dtype=torch.bool).to(device)
    imask[:, 0] = True

    for mode in ("residuals_only", "mv_concat"):
        for pretrained in (False, True):
            model = ResNet50Detector(
                num_classes=NUM_CLASSES, num_queries=NUM_QUERIES,
                input_mode=mode, pretrained=pretrained,
            ).to(device)

            boxes, classes = model(mv, res, imask)
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[{mode:16s}  pretrained={str(pretrained):5s}]  "
                  f"boxes {tuple(boxes.shape)}  classes {tuple(classes.shape)}  "
                  f"params: {n:,}")