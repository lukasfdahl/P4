import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from data_classes import Frame


GRID_H  = 120   # Frame height in macroblocks (1920x1080 / 16 rounded)
GRID_W  = 68    # Frame width  in macroblocks
FRAME_W = 1920  # Frame width  in pixels (used for MV normalization)
FRAME_H = 1080  # Frame height in pixels (used for MV normalization)


class MotionVectorDataset(Dataset):
    """
    Yields (motion_sequence, bboxes, classes) for consecutive frame windows.

    motion_sequence : (seq_len, 2, GRID_H, GRID_W)  float32, values in [-1, 1]
    bboxes          : (N, 4)  float32  [x, y, w, h] normalized
    classes         : (N,)    int64    class indices
    """

    def __init__(self, frames: list[Frame], seq_len: int = 5):
        self.frames  = frames
        self.seq_len = seq_len

    def __len__(self):
        return len(self.frames) - self.seq_len + 1

    def __getitem__(self, idx):
        window = self.frames[idx : idx + self.seq_len]

        grids = [self._to_grid(f) for f in window]
        motion_seq = torch.tensor(np.stack(grids))          # (seq_len, 2, H, W)

        last    = window[-1]
        bboxes  = torch.tensor(last.true_bounding_boxes or [], dtype=torch.float32)
        classes = torch.tensor(last.true_classes or [],         dtype=torch.long)

        return motion_seq, bboxes, classes

    def _to_grid(self, frame: Frame) -> np.ndarray:
        grid = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)
        for mv in frame.motion_vectors:
            x = min(max(mv.source_x // 16, 0), GRID_W - 1)
            y = min(max(mv.source_y // 16, 0), GRID_H - 1)
            grid[0, y, x] = (mv.motion_x / mv.motion_scale) / FRAME_W
            grid[1, y, x] = (mv.motion_y / mv.motion_scale) / FRAME_H
        return grid


def collate(batch):
    """Pad variable-length object lists so a batch can be stacked."""
    seqs, bboxes_list, classes_list = zip(*batch)

    max_obj = max((len(b) for b in bboxes_list), default=1)

    bboxes_pad  = torch.zeros(len(batch), max_obj, 4)
    classes_pad = torch.full((len(batch), max_obj), -1, dtype=torch.long)

    for i, (bb, cc) in enumerate(zip(bboxes_list, classes_list)):
        if len(bb) > 0:
            bboxes_pad[i,  :len(bb)] = bb
            classes_pad[i, :len(cc)] = cc

    return torch.stack(seqs), bboxes_pad, classes_pad


def get_loaders(frames, batch_size=16, seq_len=5, train_split=0.8, num_workers=4):
    """Split frames into train/val and return DataLoaders."""
    dataset = MotionVectorDataset(frames, seq_len=seq_len)

    n_train = int(len(dataset) * train_split)
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate, pin_memory=True)

    return train_loader, val_loader

        # Precompute valid sequence starts (need at least temporal_window frames)
        self.valid_indices = list(range(len(frames) - temporal_window + 1))
    
    def __len__(self) -> int:
        """Number of sequences in dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            motion_sequence: (temporal_window, 2, grid_height, grid_width)
            bounding_boxes: (num_objects, 4) - normalized [x, y, w, h]
            class_ids: (num_objects,) - class indices
        """
        start_idx = self.valid_indices[idx]
        sequence = self.frames[start_idx : start_idx + self.temporal_window]
        
        # Extract motion vectors for sequence
        motion_sequence = []
        for frame in sequence:
            mv_grid = self._motion_vectors_to_grid(frame.motion_vectors)
            motion_sequence.append(mv_grid)
        
        motion_sequence = np.stack(motion_sequence, axis=0)  # (seq_len, 2, H, W)
        motion_sequence = torch.from_numpy(motion_sequence).float()
        
        if self.normalize_motion:
            motion_sequence = torch.clamp(motion_sequence / 255.0, -1.0, 1.0)
        
        # Use ground truth from last frame in sequence
        last_frame = sequence[-1]
        bounding_boxes = torch.tensor(last_frame.true_bounding_boxes, dtype=torch.float32)
        class_ids = torch.tensor(last_frame.true_classes, dtype=torch.long)
        
        return motion_sequence, bounding_boxes, class_ids
    
    def _motion_vectors_to_grid(self, motion_vectors) -> np.ndarray:
        """
        Convert motion vector list to grid format.
        
        Args:
            motion_vectors: List of MotionVector objects from mv-extractor
            
        Returns:
            Grid of shape (2, grid_height, grid_width) with x, y components
        """
        grid = np.zeros((2, self.grid_height, self.grid_width), dtype=np.float32)
        
        for mv in motion_vectors:
            # Get macroblock position (already in grid coordinates)
            x = mv.source_x // 16  # Convert pixel to macroblock coordinates
            y = mv.source_y // 16
            
            # Clamp to grid bounds
            x = min(max(0, x), self.grid_width - 1)
            y = min(max(0, y), self.grid_height - 1)
            
            # Store motion components
            grid[0, y, x] = mv.motion_x
            grid[1, y, x] = mv.motion_y
        
        return grid


def create_data_loaders(
    frames: List[Frame],
    batch_size: int = 16,
    temporal_window: int = 5,
    train_split: float = 0.8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        frames: List of all Frame objects
        batch_size: Batch size for training
        temporal_window: Number of frames per sequence
        train_split: Fraction of data for training
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Split frames into train/val
    num_frames = len(frames)
    split_idx = int(num_frames * train_split)
    
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]
    
    # Create datasets
    train_dataset = CompressedVideoDataset(
        train_frames,
        temporal_window=temporal_window,
    )
    val_dataset = CompressedVideoDataset(
        val_frames,
        temporal_window=temporal_window,
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


class BatchCollator:
    """
    Custom collate function for batches with variable number of objects.
    
    Handles padding of bounding boxes to fixed size within each batch.
    """
    
    def __init__(self, max_objects: int = 50):
        """
        Initialize collator.
        
        Args:
            max_objects: Maximum objects per frame (for padding)
        """
        self.max_objects = max_objects
    
    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate batch with variable-length object lists.
        
        Returns:
            motion_sequences: (batch_size, seq_len, 2, H, W)
            bboxes_padded: (batch_size, max_objects, 4)
            classes_padded: (batch_size, max_objects)
        """
        motion_sequences, bboxes_list, classes_list = zip(*batch)
        
        motion_sequences = torch.stack(motion_sequences)
        
        # Pad bounding boxes and classes
        bboxes_padded = torch.zeros(len(batch), self.max_objects, 4)
        classes_padded = torch.full((len(batch), self.max_objects), -1, dtype=torch.long)
        
        for i, (bboxes, classes) in enumerate(zip(bboxes_list, classes_list)):
            num_objects = min(len(bboxes), self.max_objects)
            bboxes_padded[i, :num_objects] = bboxes[:num_objects]
            classes_padded[i, :num_objects] = classes[:num_objects]
        
        return motion_sequences, bboxes_padded, classes_padded
