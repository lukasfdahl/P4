"""
benchmark_model.py  –  ResNet-50 baseline for comparison against ObjectDetector.

Input modes
-----------
  "residuals_only"  (3 ch)  — pixels only, ResNet's natural habitat.
  "mv_concat"       (7 ch)  — residuals + motion vectors concatenated.
                              ResNet's conv1 is replaced with a 7-channel
                              version.  When pretrained=True the RGB weights
                              are kept for the residual channels and the 4 MV
                              channels are initialised to zero, so training
                              starts from a sensible point.

Pretrained flag
---------------
  pretrained=False  — train from scratch, fairest architectural comparison.
  pretrained=True   — ImageNet init, useful as a "best RGB baseline" ceiling.

Temporal fusion
---------------
  Residuals: mean-averaged across the clip (T frames → 1 frame).
  MVs:       I-frame MVs are zeroed (matching ObjectDetector's convention)
             then mean-averaged across frames and upsampled to the full frame
             resolution via bilinear interpolation before concatenation.

Drop-in usage in train.py
--------------------------
  # from model import ObjectDetector
  # model = ObjectDetector(...)
  from benchmark_model import ResNet50Detector
  model = ResNet50Detector(num_classes=..., num_queries=...,
                           input_mode="mv_concat", pretrained=False)
  # everything else (loss, optimizer, dataloader, eval) unchanged
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights


# ── Shared MLP (identical to model.py so heads are comparable) ───────────────

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
        hidden_dim:      int,
        num_classes:     int,
        bbox_head_dim:   int = 256,
        class_head_dim:  int = 256,
        bbox_num_layers: int = 3,
        class_num_layers:int = 3,
    ):
        super().__init__()
        self.bbox_head  = MLP(hidden_dim, bbox_head_dim,  4,           bbox_num_layers)
        self.class_head = MLP(hidden_dim, class_head_dim, num_classes, class_num_layers)

    def forward(self, queries: torch.Tensor):
        boxes   = torch.sigmoid(self.bbox_head(queries))   # [B, Q, 4]
        classes = self.class_head(queries)                  # [B, Q, num_classes]
        return boxes, classes


# ── Benchmark model ───────────────────────────────────────────────────────────

class ResNet50Detector(nn.Module):
    """
    ResNet-50 backbone → global average pool → num_queries object queries.

    Forward signature mirrors ObjectDetector exactly:
        forward(motion_vectors, residuals, iframe_mask) -> (boxes, classes)

    Parameters
    ----------
    num_classes : int
        Number of *object* classes (no-object class added automatically).
    num_queries : int
        Number of detection slots — use the same value as ObjectDetector.
    hidden_dim : int
        Width of the query projection and prediction heads.
    input_mode : str
        "residuals_only"  — 3-channel input, MVs ignored.
        "mv_concat"       — 7-channel input (3 residual + 4 MV).
    pretrained : bool
        Load ImageNet weights.  For mv_concat the RGB weights seed the
        residual channels; MV channels start at zero.
        Set False for a fair from-scratch architectural comparison.
    base_mv_scale : int
        Native H.264 MV block size (default 16).  Used to upsample MVs to
        the full frame resolution before concatenation.
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

        # ── Build backbone ────────────────────────────────────────────────────
        weights  = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        if input_mode == "mv_concat":
            # Replace conv1 to accept 7 channels (3 residual + 4 MV).
            # Copy pretrained RGB weights into the first 3 slices;
            # zero-init the 4 MV slices so the model starts from a stable point.
            old_conv = backbone.conv1                   # [64, 3, 7, 7]
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
                new_conv.weight[:, 3:, :, :] = 0.0               # MV slice — learn from scratch
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            backbone.conv1 = new_conv

        # Strip the classification head; keep everything up to avgpool
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,   # → [B, 2048, 1, 1]
        )
        self.backbone_dim = 2048

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ── Query projection ──────────────────────────────────────────────────
        # Maps the single global feature vector to Q independent query vectors
        self.query_proj = nn.Linear(self.backbone_dim, num_queries * hidden_dim)

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
        mv = mv_flat.view(B, T, *motion_vectors.shape[2:])   # [B, T, 4, H_mv, W_mv]

        # ── Temporal fusion: mean across clip ─────────────────────────────────
        res_mean = residuals.mean(dim=1)    # [B, 3, H, W]
        mv_mean  = mv.mean(dim=1)           # [B, 4, H_mv, W_mv]

        # ── Build input tensor ────────────────────────────────────────────────
        res_clamped = res_mean.clamp(0.0, 1.0)

        if self.input_mode == "residuals_only":
            x = res_clamped                 # [B, 3, H, W]

        else:  # mv_concat
            # Upsample MVs from coarse MV grid (H/16 × W/16) to full frame
            # resolution so they can be channel-concatenated with the residuals.
            mv_upsampled = F.interpolate(
                mv_mean.float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )                               # [B, 4, H, W]
            x = torch.cat([res_clamped, mv_upsampled], dim=1)  # [B, 7, H, W]

        # ── Backbone ─────────────────────────────────────────────────────────
        feat = self.backbone(x)             # [B, 2048, 1, 1]
        feat = feat.flatten(1)              # [B, 2048]

        # ── Query projection ─────────────────────────────────────────────────
        queries = self.query_proj(feat)                              # [B, Q*D]
        queries = queries.view(B, self.num_queries, self.hidden_dim) # [B, Q, D]

        # ── Prediction heads ─────────────────────────────────────────────────
        boxes, classes = self.prediction_heads(queries)
        # boxes:   [B, Q, 4]
        # classes: [B, Q, num_classes + 1]
        return boxes, classes


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BATCH       = 4
    T           = 5
    H, W        = 64, 64
    BASE_MV     = 16
    NUM_CLASSES = 10
    NUM_QUERIES = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mv    = torch.randn(BATCH, T, 4, H // BASE_MV, W // BASE_MV)
    res   = torch.rand (BATCH, T, 3, H, W)
    imask = torch.zeros(BATCH, T, dtype=torch.bool)
    imask[:, 0] = True   # first frame is always I-frame

    for mode in ("residuals_only", "mv_concat"):
        for pretrained in (False, True):
            model = ResNet50Detector(
                num_classes   = NUM_CLASSES,
                num_queries   = NUM_QUERIES,
                hidden_dim    = 256,
                input_mode    = mode,
                pretrained    = pretrained,
                base_mv_scale = BASE_MV,
            ).to(device)

            boxes, classes = model(mv.to(device), res.to(device), imask.to(device))

            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"[{mode:16s}  pretrained={str(pretrained):5s}]  "
                f"boxes {tuple(boxes.shape)}  classes {tuple(classes.shape)}  "
                f"trainable params: {n_train:,}"
            )