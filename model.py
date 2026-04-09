import torch
import torch.nn as nn
from data_classes import Frame
from torchinfo import summary


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
    Without this the transformer treats all patches as an unordered set with no
    concept of which patch is top-left vs bottom-right.
    """
    def __init__(self, hidden_dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even"
        half = hidden_dim // 2

        dim_t = torch.arange(half, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / half)

        pos_h = torch.arange(max_h, dtype=torch.float32).unsqueeze(1) / dim_t
        pos_w = torch.arange(max_w, dtype=torch.float32).unsqueeze(1) / dim_t

        pos_h[:, 0::2] = torch.sin(pos_h[:, 0::2])
        pos_h[:, 1::2] = torch.cos(pos_h[:, 1::2])
        pos_w[:, 0::2] = torch.sin(pos_w[:, 0::2])
        pos_w[:, 1::2] = torch.cos(pos_w[:, 1::2])

        self.register_buffer("pos_h", pos_h)  # [max_h, half]
        self.register_buffer("pos_w", pos_w)  # [max_w, half]

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        row_enc = self.pos_h[:h].unsqueeze(1).expand(h, w, -1)  # [H, W, half]
        col_enc = self.pos_w[:w].unsqueeze(0).expand(h, w, -1)  # [H, W, half]
        pos = torch.cat([row_enc, col_enc], dim=-1)             # [H, W, hidden_dim]
        pos = pos.reshape(1, h * w, -1)                         # [1, H*W, hidden_dim]
        return tokens + pos


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal 1D temporal positional encoding.
    Adds a unique signal per frame index so the model knows which frame in the
    clip each token came from. This is how the model learns that motion vectors
    describe movement *between* frames rather than treating each frame in isolation.
    """
    def __init__(self, hidden_dim: int, max_frames: int = 32):
        super().__init__()

        dim_t = torch.arange(hidden_dim, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim)

        pos = torch.arange(max_frames, dtype=torch.float32).unsqueeze(1) / dim_t  # [max_frames, hidden_dim]
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        self.register_buffer("pos", pos)  # [max_frames, hidden_dim]

    def forward(self, tokens: torch.Tensor, frame_idx: int) -> torch.Tensor:
        # tokens: [B, spatial_seq_len, hidden_dim]
        # Broadcast the same frame-level encoding across all spatial tokens of this frame
        temporal_enc = self.pos[frame_idx].reshape(1, 1, -1)  # [1, 1, hidden_dim]
        return tokens + temporal_enc


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
        max_h: int = 64,
        max_w: int = 64,
        max_frames: int = 32,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.spatial_pos  = PositionalEncoding2D(hidden_dim, max_h, max_w)
        self.temporal_pos = TemporalPositionalEncoding(hidden_dim, max_frames)

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

        # Add spatial + temporal positional encodings to each frame's tokens,
        # then concatenate into one long sequence: [B, T*H*W, hidden_dim]
        encoded_frames = []
        for t, tokens in enumerate(frame_tokens):
            tokens = self.spatial_pos(tokens, h, w)   # inject where in the frame
            tokens = self.temporal_pos(tokens, t)      # inject which frame in the clip
            encoded_frames.append(tokens)

        all_tokens = torch.cat(encoded_frames, dim=1)  # [B, T*H*W, hidden_dim]

        # Encoder — every token across all frames attends to every other token
        memory = self.encoder(all_tokens)              # [B, T*H*W, hidden_dim]

        # Expand object queries to the full batch
        queries = self.object_queries.weight                # [num_queries, hidden_dim]
        queries = queries.unsqueeze(0).expand(B, -1, -1)   # [B, num_queries, hidden_dim]

        # Decoder — each query cross-attends to the full spatio-temporal memory
        query_output = self.decoder(queries, memory)        # [B, num_queries, hidden_dim]

        return query_output


# the prediction heads that take the transformer's outputs and use them to predict a bounding box and a class
class PredictionHeads(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.bbox_head  = nn.Linear(hidden_dim, 4)
        self.class_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, transformer_output):
        # Sigmoid constrains box coordinates to [0, 1] (normalised image coordinates)
        boxes   = torch.sigmoid(self.bbox_head(transformer_output))
        classes = self.class_head(transformer_output)
        return boxes, classes


# the full transformer model with the backbone core and head all combined.
class ObjectDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        block_size: int = 16,
        clip_length: int = 5,       # number of consecutive frames the model sees at once
        expected_h_tokens: int = 4, # used to size num_queries; update if your resolution changes
        expected_w_tokens: int = 4,
    ):
        super().__init__()
        self.block_size  = block_size
        self.clip_length = clip_length

        # num_queries scales with the number of tokens so the decoder is neither
        # starved (fewer queries than tokens) nor massively over-provisioned.
        num_queries = clip_length * expected_h_tokens * expected_w_tokens

        self.tokenizer = H264Tokenizer(4, 3, 256, block_size)
        self.transformer = TransformerCore(
            hidden_dim=256,
            num_heads=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            num_queries=num_queries,
            max_h=64,
            max_w=64,
            max_frames=clip_length,
        )
        self.prediction_heads = PredictionHeads(hidden_dim=256, num_classes=num_classes)

    def forward(
        self,
        motion_vectors: torch.Tensor,  # [B, T, 4, H_tokens, W_tokens]
        residuals: torch.Tensor,       # [B, T, 3, H, W]
        frame_types: list[str],        # length T, e.g. ['I', 'P', 'P', 'B', 'P']
    ):
        T = motion_vectors.size(1)
        H, W = residuals.shape[3], residuals.shape[4]
        h_tokens = H // self.block_size
        w_tokens = W // self.block_size

        # Tokenize each frame independently, then pass the list to the transformer
        frame_tokens = []
        for t in range(T):
            is_iframe = (frame_types[t] == 'I')
            tokens = self.tokenizer(motion_vectors[:, t], residuals[:, t], is_iframe)
            frame_tokens.append(tokens)

        transformer_output = self.transformer(frame_tokens, h_tokens, w_tokens)
        final_boxes, final_classes = self.prediction_heads(transformer_output)

        return final_boxes, final_classes

class WrappedModel(nn.Module):
    def __init__(self, model, frame_types):
        super().__init__()
        self.model = model
        self.frame_types = frame_types

    def forward(self, motion_vectors, residuals):
        return self.model(motion_vectors, residuals, self.frame_types)

# helper for dummy data — generates a full clip of T frames
def _generate_dummy_data(batch_size, clip_length=5, h_tokens=4, w_tokens=4, block_size=16):
    # [B, T, 4, H_tokens, W_tokens]
    motion_vectors = torch.randn(batch_size, clip_length, 4, h_tokens, w_tokens)
    # [B, T, 3, H, W]
    residuals = torch.randn(batch_size, clip_length, 3, h_tokens * block_size, w_tokens * block_size)
    # Realistic frame type sequence: first frame is always I, rest are P/B
    frame_types = ['I'] + ['P'] * (clip_length - 1)
    return motion_vectors, residuals, frame_types


if __name__ == "__main__":
    CLIP_LENGTH    = 5
    H_TOKENS       = 4
    W_TOKENS       = 4
    BLOCK_SIZE     = 16

    model = ObjectDetector(
        num_classes=10,
        block_size=BLOCK_SIZE,
        clip_length=CLIP_LENGTH,
        expected_h_tokens=BLOCK_SIZE,
        expected_w_tokens=BLOCK_SIZE,
    )

    motion_vectors, residuals, frame_types = _generate_dummy_data(batch_size=4)



    wrapped_model = WrappedModel(model, frame_types)

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