import torch
import torch.nn as nn
from data_classes import Frame
from torchinfo import summary
from torchview import draw_graph

import torch.nn.functional as F


class MultiScaleH264Tokenizer(nn.Module):
    """
    Tokenizes each frame at multiple block sizes simultaneously, then fuses
    them with a small learned gate.
 
    Scales and their token grids for a frame:
        scale  8  →  8×8  = 64 tokens  (finest, most spatial detail)
        scale 16  →  4×4  = 16 tokens  (native H.264 MV resolution)
        scale 32  →  2×2  =  4 tokens  (coarsest, global motion cues)
 
    After per-scale projection, coarser grids are bilinearly upsampled to the
    finest resolution so all three can be added together.
 
    Learned gate
    `scale_gate` is a length-3 parameter vector (one scalar per scale).
    At forward time it passes through softmax → three non-negative weights
    that sum to 1.  Initialised to zeros so training starts with equal weight
    on all scales and specialises from there.
 
    At scale 8 the MVs are upsampled (H.264 doesn't measure 8×8 block MVs
    natively, it is 16x16).  The model learns to rely on residuals more at fine scales and
    on MVs more at scale 16.
 
    Args:
        mv_channels       : number of MV feature channels (4: source, x, y, scale)
        residual_channels : colour channels of the residual (3: YUV or RGB)
        hidden_dim        : output channel depth after projection
        scales            : list of block sizes to process (default [8, 16, 32])
        base_mv_scale     : the block size H.264 actually uses for MVs (16).
                            MVs are resampled from this grid to each target scale.
    """
 
    def __init__(
        self,
        mv_channels:       int       = 4,
        residual_channels: int       = 3,
        hidden_dim:        int       = 256,
        scales:            list[int] = [8, 16, 32],
        base_mv_scale:     int       = 16,
    ):
        super().__init__()
        self.scales       = sorted(scales)      # ascending: finest → coarsest
        self.finest_scale = self.scales[0]
        self.base_mv_scale = base_mv_scale
 
        total_channels = mv_channels + residual_channels
 
        # One Conv2d per scale to downsample residuals to that scale's token grid.
        # kernel_size == stride == block_size does a non-overlapping spatial pooling.
        self.residual_downsamplers = nn.ModuleList([
            nn.Conv2d(residual_channels, residual_channels, kernel_size=s, stride=s)
            for s in self.scales
        ])
 
        # One 1×1 conv per scale to project (mv_ch + res_ch) → hidden_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(total_channels, hidden_dim, kernel_size=1)
            for _ in self.scales
        ])
 
        # Learned global gate — one scalar per scale.
        # Zeros → uniform softmax at initialisation (equal weight to all scales).
        self.scale_gate = nn.Parameter(torch.zeros(len(self.scales)))
 
    def forward(
        self,
        motion_vectors: torch.Tensor,   # [B, 4, H_mv, W_mv]   (at base_mv_scale grid)
        residuals:      torch.Tensor,   # [B, 3, H, W]
        is_iframe:      bool = False,
    ) -> torch.Tensor:
        
        # Returns [B, H_finest * W_finest, hidden_dim]

        # check if i-frame
        # I-frames have no meaningful motion; zero them out rather than passing noise.
        if is_iframe:
            motion_vectors = torch.zeros_like(motion_vectors)
 
        gate_weights = torch.softmax(self.scale_gate, dim=0)   # [num_scales]  sums to 1

        # Get the spatial dimensions of the residuals
        H, W      = residuals.shape[2], residuals.shape[3]
        finest_h  = H // self.finest_scale
        finest_w  = W // self.finest_scale

        # fused tensor to accumulate features across scales
        fused: torch.Tensor | None = None

        # Iterate over each scale
        for i, scale in enumerate(self.scales):
            target_h = H // scale
            target_w = W // scale
 
            # residuals at this scale
            # Conv2d with kernel=stride=scale does non-overlapping spatial pooling
            res_down = self.residual_downsamplers[i](residuals)     # [B, 3, H/s, W/s]
 
            # motion vectors resampled to this scale's grid
            # bilinear gives smoother upsampling (scale<16) and sensible downsampling (scale>16)
            mv_at_scale = F.interpolate(
                motion_vectors.float(),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            )                                                        # [B, 4, H/s, W/s]
 
            # concat channels then project to hidden_dim
            combined = torch.cat([mv_at_scale, res_down], dim=1)   # [B, 7, H/s, W/s]
            tokens   = self.projections[i](combined)                # [B, hidden_dim, H/s, W/s]
 
            # upsample coarser scales to finest resolution before fusion
            if scale != self.finest_scale:
                tokens = F.interpolate(
                    tokens,
                    size=(finest_h, finest_w),
                    mode='bilinear',
                    align_corners=False,
                )                                                    # [B, hidden_dim, finest_h, finest_w]
 
            # weighted accumulation
            fused = gate_weights[i] * tokens if fused is None else fused + gate_weights[i] * tokens
 
        # [B, hidden_dim, finest_h, finest_w] → [B, finest_h * finest_w, hidden_dim]
        return fused.flatten(2).permute(0, 2, 1)
 

# old tokenizer (non multi)
# tokenize the data for the transformer model (block size is the size of the blocks used for the motion vectors in the h264 video)
class H264Tokenizer(nn.Module):
    def __init__(self, mv_channels, residual_channels, hidden_dim, block_size=16):
        super().__init__()

        # Downsamples the residuals to match the motion vector spatial grid
        # e.g. 64x64 residuals → 4x4 with block_size=16
        self.residual_downsample = nn.Conv2d(
            in_channels=residual_channels,
            out_channels=residual_channels,
            kernel_size=block_size,
            stride=block_size,
        )

        # Projects the combined MV + residual channels to hidden_dim
        total_input_channels = mv_channels + residual_channels
        self.projection = nn.Conv2d(total_input_channels, hidden_dim, kernel_size=1)

    def forward(self, motion_vectors: torch.Tensor, residuals: torch.Tensor, is_iframe: bool = False) -> torch.Tensor:
        # I-frames have no meaningful motion vectors (they are keyframes with no reference
        # to a previous frame). Zero them out so they don't inject noise into the model.
        if is_iframe:
            motion_vectors = torch.zeros_like(motion_vectors)

        # Scale down residuals to match the MV token grid
        small_residuals = self.residual_downsample(residuals)

        # Concatenate MV channels and downsampled residual channels
        combined_features = torch.cat([motion_vectors, small_residuals], dim=1)

        # Project to hidden_dim channels
        tokens = self.projection(combined_features)

        # Flatten spatial grid → sequence: [B, hidden_dim, H, W] → [B, H*W, hidden_dim]
        tokens = tokens.flatten(2).permute(0, 2, 1)
        return tokens


class PositionalEncoding2D(nn.Module):
    """
    Sinusoidal 2D spatial positional encoding.
    Computed once per unique (h, w) grid and cached as a GPU tensor.
    Subsequent calls with the same grid size are a single in-place add — no recompute.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even"
        self.hidden_dim = hidden_dim
        self.half_dim = hidden_dim // 2
        # Cache: (h, w) -> [1, h*w, hidden_dim] float32 tensor on GPU
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def _build(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        half = self.half_dim
        dim_t = torch.arange(half, device=device, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / half)

        pos_h = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1) / dim_t
        pos_w = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1) / dim_t

        pos_h[:, 0::2] = torch.sin(pos_h[:, 0::2])
        pos_h[:, 1::2] = torch.cos(pos_h[:, 1::2])
        pos_w[:, 0::2] = torch.sin(pos_w[:, 0::2])
        pos_w[:, 1::2] = torch.cos(pos_w[:, 1::2])

        row_enc = pos_h.unsqueeze(1).expand(h, w, -1)
        col_enc = pos_w.unsqueeze(0).expand(h, w, -1)
        pos = torch.cat([row_enc, col_enc], dim=-1)    # [h, w, hidden_dim]
        return pos.reshape(1, h * w, self.hidden_dim)  # [1, h*w, hidden_dim]

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        tokens: [B, h*w, hidden_dim]
        returns: [B, h*w, hidden_dim]
        """
        key = (h, w)
        if key not in self._cache:
            self._cache[key] = self._build(h, w, tokens.device)
        return tokens + self._cache[key].to(dtype=tokens.dtype)


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal 1D temporal positional encoding.
    Computed once per unique frame_idx and cached on GPU.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Cache: frame_idx -> [1, 1, hidden_dim] float32 tensor on GPU
        self._cache: dict[int, torch.Tensor] = {}

    def _build(self, frame_idx: int, device: torch.device) -> torch.Tensor:
        dim_t = torch.arange(self.hidden_dim, device=device, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.hidden_dim)
        pos = torch.tensor([frame_idx], device=device, dtype=torch.float32).unsqueeze(1) / dim_t.unsqueeze(0)
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        return pos.reshape(1, 1, self.hidden_dim)  # [1, 1, hidden_dim]

    def forward(self, tokens: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        tokens: [B, spatial_seq_len, hidden_dim]
        returns: [B, spatial_seq_len, hidden_dim]
        """
        if frame_idx not in self._cache:
            self._cache[frame_idx] = self._build(frame_idx, tokens.device)
        return tokens + self._cache[frame_idx].to(dtype=tokens.dtype)


# the actual core of the transformer model
class TransformerCore(nn.Module):
    """
    DETR-style encoder-decoder transformer. See https://arxiv.org/abs/2005.12872

    Encoder:  self-attention across ALL tokens from ALL frames in the clip.
              Every spatial patch can attend to every other patch across time,
              giving the model full spatio-temporal context.

    Decoder:  N learned object queries cross-attend to the encoder output.
              Each query specialises during training to detect objects of
              certain sizes/positions/motion patterns.

    Output:   [B, num_queries, hidden_dim] — one vector per object candidate,
              fed straight into PredictionHeads.

    num_queries should be set to roughly clip_length * h_tokens * w_tokens
    so there are enough queries to cover the token sequence without being
    vastly over- or under-provisioned relative to the input.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_queries: int = 80,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.spatial_pos  = PositionalEncoding2D(hidden_dim)
        self.temporal_pos = TemporalPositionalEncoding(hidden_dim)

        # Encoder: self-attention + FFN across all spatio-temporal tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Object queries: learned vectors, one per candidate detection
        self.object_queries = nn.Embedding(num_queries, hidden_dim)

        # Decoder: queries cross-attend to encoder memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, frame_tokens: list[torch.Tensor], h: int, w: int) -> torch.Tensor:
        """
        Args:
            frame_tokens : list of T tensors, each [B, H*W, hidden_dim] — one per frame
            h, w         : spatial token grid dimensions
        Returns:
            [B, num_queries, hidden_dim]
        """
        B = frame_tokens[0].size(0)
        T = len(frame_tokens)

        # Stack to [B, T, S, D] then add positional encodings in two vectorised ops
        # instead of T sequential Python loop iterations.
        all_tokens = torch.stack(frame_tokens, dim=1)  # [B, T, S, D]

        # Spatial encoding: [1, 1, S, D] — same for every frame, broadcast over B and T
        sp_key = (h, w)
        if sp_key not in self.spatial_pos._cache:
            self.spatial_pos._cache[sp_key] = self.spatial_pos._build(h, w, all_tokens.device)
        sp_enc = self.spatial_pos._cache[sp_key].unsqueeze(1)  # [1, 1, S, D]
        all_tokens = all_tokens + sp_enc.to(dtype=all_tokens.dtype)

        # Temporal encoding: [1, T, 1, D] — build all T vectors at once, broadcast over B and S
        temp_encs = []
        for t in range(T):
            if t not in self.temporal_pos._cache:
                self.temporal_pos._cache[t] = self.temporal_pos._build(t, all_tokens.device)
            temp_encs.append(self.temporal_pos._cache[t])  # each [1, 1, D]
        # [1, T, 1, D]
        temp_enc = torch.cat(temp_encs, dim=1).to(dtype=all_tokens.dtype)
        all_tokens = all_tokens + temp_enc  # broadcasts over B and S

        # Flatten to [B, T*S, D] for the encoder
        all_tokens = all_tokens.flatten(1, 2)

        # Encoder — every token across all frames attends to every other token
        memory = self.encoder(all_tokens)              # [B, T*H*W, hidden_dim]

        # Expand object queries to the full batch
        queries = self.object_queries.weight                # [num_queries, hidden_dim]
        queries = queries.unsqueeze(0).expand(B, -1, -1)   # [B, num_queries, hidden_dim]

        # Decoder — each query cross-attends to the full spatio-temporal memory
        query_output = self.decoder(queries, memory)        # [B, num_queries, hidden_dim]

        return query_output

# implement mlp block for both heads
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

# the prediction heads that take the transformer's outputs and use them to predict a bounding box and a class
class PredictionHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        bbox_head_dim: int = 256,
        class_head_dim: int = 256,
        bbox_num_layers: int = 3,
        class_num_layers: int = 3,
    ):
        super().__init__()

        self.bbox_head = MLP(
            input_dim=hidden_dim,
            hidden_dim=bbox_head_dim,
            output_dim=4,
            num_layers=bbox_num_layers,
        )

        self.class_head = MLP(
            input_dim=hidden_dim,
            hidden_dim=class_head_dim,
            output_dim=num_classes,
            num_layers=class_num_layers,
        )

    def forward(self, transformer_output: torch.Tensor):
        boxes = torch.sigmoid(self.bbox_head(transformer_output))
        classes = self.class_head(transformer_output)
        return boxes, classes

# the full transformer model with the backbone core and head all combined.
class ObjectDetector(nn.Module):
    """
    Full detection model: multi-scale tokenizer → DETR transformer → prediction heads.
 
    Args:
        num_classes        : number of detection classes
        scales             : block sizes for multi-scale tokenization (default [8, 16, 32]).
                             The finest scale (min(scales)) determines the output token grid.
        base_mv_scale      : native H.264 MV block size (almost always 16 — don't change
                             unless you re-encode your video with a different block size).
        clip_length        : number of consecutive frames per forward pass
        expected_h_tokens  : H // min(scales),  used to size num_queries at init time.
                             For a 64×64 frame with scales=[8,16,32]:  64 // 8 = 8.
        expected_w_tokens  : W // min(scales),  same rule.
    """

    def __init__(
        self,
        num_classes: int       = 10,
        scales: list[int] | None = None,
        base_mv_scale: int     = 16,
        clip_length: int       = 5,
        expected_h_tokens: int = 8,   # 64 // 8
        expected_w_tokens: int = 8,   # 64 // 8
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        ffn_dim: int | None = None,
    ):
        super().__init__()  

        # default scales
        if scales is None:
            scales = [8, 16, 32]

        # ffn should be 4 times hidden_dim because of the way the transformer architecture expands channels in the feedforward layers
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim


        self.scales       = sorted(scales)
        self.finest_scale = min(scales)   # token grid resolution used by the transformer
        self.clip_length  = clip_length

        # num_queries scales with the number of tokens so the decoder is neither
        # starved (fewer queries than tokens) nor massively over-provisioned.
        num_queries = 10 # fixed

        # now with multiscale
        self.tokenizer = MultiScaleH264Tokenizer(
            mv_channels=4,
            residual_channels=3,
            hidden_dim=hidden_dim,
            scales=scales,
            base_mv_scale=base_mv_scale,
        )

        self.transformer = TransformerCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            ffn_dim=ffn_dim,
            num_queries=num_queries,
        )

        self.prediction_heads = PredictionHeads(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            bbox_head_dim=hidden_dim,
            class_head_dim=hidden_dim,
            bbox_num_layers=3,
            class_num_layers=3,
        )

    def forward(
        self,
        motion_vectors: torch.Tensor,  # [B, T, 4, H_tokens, W_tokens]
        residuals: torch.Tensor,       # [B, T, 3, H, W]
        frame_types: list[str],        # length T, e.g. ['I', 'P', 'P', 'B', 'P']
    ):
        B, T = motion_vectors.shape[:2]
        H, W = residuals.shape[3], residuals.shape[4]
        h_tokens = H // self.finest_scale
        w_tokens = W // self.finest_scale

        # Fold T into batch so the tokenizer runs ONE forward pass instead of T.
        # [B, T, C, H, W] -> [B*T, C, H, W]
        mv_flat  = motion_vectors.view(B * T, *motion_vectors.shape[2:])
        res_flat = residuals.view(B * T, *residuals.shape[2:])

        # Zero I-frame MVs in one vectorised mask instead of a per-frame branch.
        # Build a [B*T] bool mask: True where the frame is an I-frame.
        iframe_mask = torch.tensor(
            [ft == 'I' for ft in frame_types],  # [T] — same sequence for every clip
            device=mv_flat.device,
        ).repeat_interleave(B)  # broadcast over batch: [B*T]
        # Reshape for broadcasting against [B*T, 4, H_mv, W_mv]
        mv_flat = mv_flat.clone()   # avoid in-place on a view
        mv_flat[iframe_mask] = 0.0

        # Single tokenizer forward: [B*T, S, D]  where S = h_tokens * w_tokens
        all_tokens = self.tokenizer(mv_flat, res_flat, is_iframe=False)

        # Unfold back to [B, T, S, D] then split into a list of T tensors [B, S, D]
        all_tokens = all_tokens.view(B, T, *all_tokens.shape[1:])
        frame_tokens = list(all_tokens.unbind(dim=1))

        transformer_output = self.transformer(frame_tokens, h_tokens, w_tokens)
        final_boxes, final_classes = self.prediction_heads(transformer_output)

        return final_boxes, final_classes


# Wrapper for torchinfo summary (hides frame_types from the signature)
class WrappedModel(nn.Module):
    def __init__(self, model, frame_types):
        super().__init__()
        self.model = model
        self.frame_types = frame_types

    def forward(self, motion_vectors, residuals):
        return self.model(motion_vectors, residuals, self.frame_types)

# helper for dummy data — generates a full clip of frames
def _generate_dummy_data(
    batch_size:    int = 4,
    clip_length:   int = 5,
    base_mv_scale: int = 16,
    frame_h:       int = 64,
    frame_w:       int = 64,
):
    h_mv = frame_h // base_mv_scale   # = 4 for 64×64 @ scale 16
    w_mv = frame_w // base_mv_scale
    
    # [B, T, 4, H_tokens, W_tokens]
    motion_vectors = torch.randn(batch_size, clip_length, 4, h_mv, w_mv)

    # [B, T, 3, H, W]
    residuals = torch.randn(batch_size, clip_length, 3, frame_h, frame_w)

    # Realistic frame type sequence: first frame is always I, rest are P/B
    frame_types = ['I'] + ['P'] * (clip_length - 1)
    return motion_vectors, residuals, frame_types


if __name__ == "__main__":
    CLIP_LENGTH      = 8
    FRAME_H          = 64
    FRAME_W          = 64
    SCALES           = [8, 16, 32]

    # expected_h/w_tokens = frame_h // finest_scale = 64 // 8 = 8
    model = ObjectDetector(
        num_classes       = 10,
        scales            = SCALES,
        base_mv_scale     = 16,
        clip_length       = CLIP_LENGTH,
        expected_h_tokens = FRAME_H // min(SCALES),   # 8
        expected_w_tokens = FRAME_W // min(SCALES),   # 8
    )
    
    # generate dummy data for testing
    motion_vectors, residuals, frame_types = _generate_dummy_data(
        batch_size  = 4,
        clip_length = CLIP_LENGTH,
        frame_h     = FRAME_H,
        frame_w     = FRAME_W,
    )

    wrapped_model = WrappedModel(model, frame_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped_model.to(device)
    motion_vectors = motion_vectors.to(device)
    residuals      = residuals.to(device)

    summary(
        wrapped_model,
        input_data=(motion_vectors, residuals),
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
    )

    print(f"Input motion_vectors: {motion_vectors.shape}")  # [4, 5, 4, 4, 4]
    print(f"Input residuals:      {residuals.shape}")       # [4, 5, 3, 64, 64]
    print(f"Frame types:          {frame_types}")

    boxes, classes = model(motion_vectors, residuals, frame_types)
    print(f"\nOutput boxes:   {boxes.shape}")    # [4, 80, 4]
    print(f"Output classes: {classes.shape}")   # [4, 80, 10]

    # Check if the output is on the same device as the model
    print(f"Output boxes device: {boxes.device}")
    print(f"Output classes device: {classes.device}")

    # learned gate weights to see initial scale distribution
    gate = torch.softmax(model.tokenizer.scale_gate, dim=0)
    for s, w in zip(SCALES, gate.tolist()):
        print(f"  scale {s:2d}  gate weight: {w:.4f}")

    graph = draw_graph(
        wrapped_model,
        input_data=(motion_vectors, residuals),
        expand_nested=True,
    )

    graph.visual_graph.render(filename="model_graph", format="png", cleanup=True)
    print("Saved model_graph.png")