from dataclasses import dataclass


@dataclass
class Config:
    num_classes:     int   = 23      # YouTube-BoundingBoxes
    seq_len:         int   = 5       # Frames per sequence fed to ConvLSTM
    hidden_channels: int   = 32      # ConvLSTM hidden state size
    batch_size:      int   = 16
    lr:              float = 1e-3
    weight_decay:    float = 1e-4
    epochs:          int   = 50
    train_split:     float = 0.8
    num_workers:     int   = 4
    checkpoint_dir:  str   = "./checkpoints"
