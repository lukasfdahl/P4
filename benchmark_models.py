"""
benchmark_models.py  –  Benchmark models for comparison against ObjectDetector.

FasterRCNNDetector
------------------
  Uses torchvision's ResNet50-FPN backbone (extracted from Faster R-CNN).
  Multi-scale FPN features are globally pooled and projected to Q query
  vectors, then fed through the same MLP prediction heads as ObjectDetector.
  The key advantage over plain ResNet50 is the FPN top-down pathway, which
  fuses multi-scale spatial information before pooling.

  Forward signature mirrors ObjectDetector exactly:
      forward(motion_vectors, residuals, iframe_mask) -> (boxes, classes)

RandomDetector
--------------
  Outputs uniformly random boxes and class logits every forward pass.
  Has a single dummy parameter (required so AdamW does not raise an empty-
  parameter error), but that parameter is never part of the computation graph
  so no gradient updates ever occur.  Serves as a hard lower-bound baseline.

Input modes (FasterRCNNDetector only)
--------------------------------------
  "residuals_only"  (3 ch) — RGB residuals only.
  "mv_concat"       (7 ch) — residuals + motion vectors concatenated.
                             The first conv of the ResNet body is replaced
                             with a 7-channel version; pretrained RGB weights
                             seed the residual channels, MV channels start
                             at zero.

Temporal fusion (FasterRCNNDetector)
-------------------------------------
  Residuals: mean-averaged across the clip (T frames → 1 frame).
  MVs:       I-frame MVs are zeroed then mean-averaged and upsampled to the
             full frame resolution before concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection as tv_det
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


# ── Shared MLP + heads (identical to model.py so heads are comparable) ────────

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        assert num_layers >= 1
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHeads(nn.Module):
    def __init__(
        self,
        hidden_dim:       int,
        num_classes:      int,
        bbox_head_dim:    int = 256,
        class_head_dim:   int = 256,
        bbox_num_layers:  int = 3,
        class_num_layers: int = 3,
    ):
        super().__init__()
        self.bbox_head  = MLP(hidden_dim, bbox_head_dim,  4,           bbox_num_layers)
        self.class_head = MLP(hidden_dim, class_head_dim, num_classes, class_num_layers)

    def forward(self, queries: torch.Tensor):
        boxes   = torch.sigmoid(self.bbox_head(queries))   # [B, Q, 4]
        classes = self.class_head(queries)                  # [B, Q, num_classes]
        return boxes, classes


# ── FasterRCNNDetector ────────────────────────────────────────────────────────

class FasterRCNNDetector(nn.Module):
    """
    ResNet50-FPN backbone (from Faster R-CNN) → global avg pool → Q queries.

    The FPN top-down pathway fuses features from all four ResNet stages into
    256-channel maps at four spatial scales.  The coarsest map ('3') is
    globally average-pooled to a single 256-d vector, then projected to Q
    independent query vectors that are passed through the same MLP heads used
    by ObjectDetector.

    Parameters
    ----------
    num_classes : int
        Number of object classes (no-object class added internally).
    num_queries : int
        Number of detection slots — use the same value as ObjectDetector.
    hidden_dim : int
        Width of the query projection and prediction heads.
    input_mode : str
        "residuals_only"  — 3-channel input, MVs ignored.
        "mv_concat"       — 7-channel input (3 residual + 4 MV).
    pretrained : bool
        Load COCO-pretrained Faster R-CNN weights for the FPN backbone.
        Set False for a fair from-scratch architectural comparison.
    base_mv_scale : int
        Native H.264 MV block size (default 16).
    freeze_backbone : bool
        Freeze all backbone parameters after construction.
    """

    _VALID_MODES = {"residuals_only", "mv_concat"}

    def __init__(
        self,
        num_classes:     int  = 10,
        num_queries:     int  = 10,
        hidden_dim:      int  = 256,
        input_mode:      str  = "residuals_only",
        pretrained:      bool = False,
        base_mv_scale:   int  = 16,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if input_mode not in self._VALID_MODES:
            raise ValueError(
                f"input_mode must be one of {self._VALID_MODES}, got '{input_mode}'"
            )

        self.input_mode         = input_mode
        self.base_mv_scale      = base_mv_scale
        self.num_queries        = num_queries
        self.num_object_classes = num_classes
        self.no_object_class    = num_classes
        self.num_output_classes = num_classes + 1
        self.hidden_dim         = hidden_dim

        # ── Build FPN backbone ────────────────────────────────────────────────
        weights  = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
        rcnn     = tv_det.fasterrcnn_resnet50_fpn(weights=weights)
        backbone = rcnn.backbone   # BackboneWithFPN: .body (ResNet) + .fpn (FPN)

        if input_mode == "mv_concat":
            # Replace first conv of the ResNet body to accept 7 channels.
            # Pretrained RGB weights seed the residual (first 3) channels;
            # the 4 MV channels are zero-initialised.
            old_conv = backbone.body.conv1              # [64, 3, 7, 7]
            new_conv = nn.Conv2d(
                in_channels  = 7,
                out_channels = old_conv.out_channels,   # 64
                kernel_size  = old_conv.kernel_size,
                stride       = old_conv.stride,
                padding      = old_conv.padding,
                bias         = old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight   # RGB slice
                new_conv.weight[:, 3:, :, :] = 0.0               # MV slice
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            backbone.body.conv1 = new_conv

        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # FPN outputs 256-channel maps at keys '0','1','2','3','pool'.
        # We global-avg-pool the coarsest full-resolution map ('3') → [B, 256].
        self.backbone_out_dim = 256

        # ── Query projection ──────────────────────────────────────────────────
        self.query_proj = nn.Linear(self.backbone_out_dim, num_queries * hidden_dim)

        # ── Prediction heads (identical architecture to ObjectDetector) ───────
        self.prediction_heads = PredictionHeads(
            hidden_dim       = hidden_dim,
            num_classes      = self.num_output_classes,
            bbox_head_dim    = hidden_dim,
            class_head_dim   = hidden_dim,
            bbox_num_layers  = 3,
            class_num_layers = 3,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        motion_vectors: torch.Tensor,   # [B, T, 4, H_mv, W_mv]
        residuals:      torch.Tensor,   # [B, T, 3, H, W]
        iframe_mask:    torch.Tensor,   # [B, T] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, _, H, W = residuals.shape

        # ── Zero I-frame MVs (mirrors ObjectDetector's convention) ────────────
        mv_flat    = motion_vectors.reshape(B * T, *motion_vectors.shape[2:]).clone()
        imask_flat = iframe_mask.to(device=mv_flat.device, dtype=torch.bool).reshape(B * T)
        mv_flat[imask_flat] = 0.0
        mv = mv_flat.view(B, T, *motion_vectors.shape[2:])

        # ── Temporal fusion: mean across clip ─────────────────────────────────
        res_mean = residuals.mean(dim=1)    # [B, 3, H, W]
        mv_mean  = mv.mean(dim=1)           # [B, 4, H_mv, W_mv]

        # ── Build input tensor ────────────────────────────────────────────────
        res_clamped = res_mean.clamp(0.0, 1.0)

        if self.input_mode == "residuals_only":
            x = res_clamped                 # [B, 3, H, W]
        else:   # mv_concat
            mv_up = F.interpolate(
                mv_mean.float(), size=(H, W), mode="bilinear", align_corners=False,
            )                               # [B, 4, H, W]
            x = torch.cat([res_clamped, mv_up], dim=1)   # [B, 7, H, W]

        # ── FPN backbone → coarsest feature map → global avg pool ─────────────
        # backbone returns OrderedDict {'0','1','2','3','pool'}
        # '3' is [B, 256, H/32, W/32]; adaptive_avg_pool2d handles any spatial size.
        features = self.backbone(x)
        feat = F.adaptive_avg_pool2d(features["3"], (1, 1)).flatten(1)  # [B, 256]

        # ── Query projection ─────────────────────────────────────────────────
        queries = self.query_proj(feat)                              # [B, Q*D]
        queries = queries.view(B, self.num_queries, self.hidden_dim) # [B, Q, D]

        # ── Prediction heads ─────────────────────────────────────────────────
        boxes, classes = self.prediction_heads(queries)
        # boxes:   [B, Q, 4]
        # classes: [B, Q, num_classes + 1]
        return boxes, classes


# ── RandomDetector ────────────────────────────────────────────────────────────

class RandomDetector(nn.Module):
    """
    Random-guessing lower-bound baseline.

    Outputs uniformly random boxes (with valid xmin<xmax, ymin<ymax) and
    random class logits every forward pass.  The dummy parameter is required
    so AdamW does not raise an empty-parameter error; it is never part of the
    computation graph so no gradient updates ever occur.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_queries: int = 10,
        **kwargs,   # absorbs unused config keys (hidden_dim, input_mode, etc.)
    ):
        super().__init__()
        self.num_queries        = num_queries
        self.num_output_classes = num_classes + 1
        self.no_object_class    = num_classes

        # Required so AdamW does not raise "optimizer got an empty parameter list".
        # Never used in forward — no gradient flows through it.
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        motion_vectors: torch.Tensor,   # [B, T, 4, H_mv, W_mv]  (ignored)
        residuals:      torch.Tensor,   # [B, T, 3, H, W]         (device source only)
        iframe_mask:    torch.Tensor,   # [B, T] bool              (ignored)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B      = residuals.shape[0]
        device = residuals.device

        # Random boxes in [0, 1]; sort each pair so xmin<=xmax, ymin<=ymax.
        raw   = torch.rand(B, self.num_queries, 4, device=device)
        boxes = torch.stack([
            torch.minimum(raw[..., 0], raw[..., 1]),   # xmin
            torch.maximum(raw[..., 0], raw[..., 1]),   # xmax
            torch.minimum(raw[..., 2], raw[..., 3]),   # ymin
            torch.maximum(raw[..., 2], raw[..., 3]),   # ymax
        ], dim=-1)

        # Random logits — softmax will pick a random class each time.
        classes = torch.randn(B, self.num_queries, self.num_output_classes, device=device)

        return boxes, classes


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BATCH       = 2
    T           = 5
    H, W        = 64, 64
    BASE_MV     = 16
    NUM_CLASSES = 5
    NUM_QUERIES = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mv    = torch.randn(BATCH, T, 4, H // BASE_MV, W // BASE_MV)
    res   = torch.rand (BATCH, T, 3, H, W)
    imask = torch.zeros(BATCH, T, dtype=torch.bool)
    imask[:, 0] = True

    print("── FasterRCNNDetector ──")
    for mode in ("residuals_only", "mv_concat"):
        for pretrained in (False,):
            model = FasterRCNNDetector(
                num_classes=NUM_CLASSES, num_queries=NUM_QUERIES,
                hidden_dim=256, input_mode=mode, pretrained=pretrained,
            ).to(device)
            boxes, classes = model(mv.to(device), res.to(device), imask.to(device))
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [{mode:16s}]  boxes {tuple(boxes.shape)}  "
                  f"classes {tuple(classes.shape)}  trainable: {n:,}")

    print("── RandomDetector ──")
    model = RandomDetector(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES).to(device)
    boxes, classes = model(mv.to(device), res.to(device), imask.to(device))
    print(f"  boxes {tuple(boxes.shape)}  classes {tuple(classes.shape)}  "
          f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
