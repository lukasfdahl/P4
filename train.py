import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import copy
import os
import numpy as np

# Import from your repository files
from eval_framework import BoundingBox, Prediction, ModelMetric, evaluate
from data_classes import Frame, MotionVector

# ---------------------------------------------------------------------------
# 1. Dataset Wrapper (From 'development' / 'residuals' branches)
# ---------------------------------------------------------------------------
class CompressedVideoDataset(Dataset):
    """
    PyTorch Dataset wrapper. 
    Merge your 'development' dataset loader and 'residuals' extraction logic here.
    """
    def __init__(self, data_path, sequence_length=5):
        self.data_path = data_path
        self.sequence_length = sequence_length
        # TODO: Load your extracted frames, motion vectors, and residuals here.
        self.samples = [] 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: Return a sequence of frames and their corresponding ground truths.
        # Expected outputs:
        # x_motion: Tensor of shape (seq_len, 2, H, W)
        # x_residual: Tensor of shape (seq_len, channels, H, W)
        # targets: Dictionary containing 'boxes' and 'classes' for the sequence
        pass


# ---------------------------------------------------------------------------
# 2. Model Architecture (From 'model' branch)
# ---------------------------------------------------------------------------
class ConvLSTMDetector(nn.Module):
    """
    The ConvLSTM temporal encoder + YOLO-style detection head 
    (as described in Section 6.4.2 of your report).
    """
    def __init__(self, num_classes=23):
        super(ConvLSTMDetector, self).__init__()
        # TODO: Paste the architecture from your 'model_diff.txt' here.
        # e.g., self.conv_lstm = ConvLSTM(...)
        # e.g., self.detection_head = YOLOHead(...)

    def forward(self, x_motion, x_residual):
        # TODO: Process the sequence and return bounding box predictions
        return torch.tensor([])


# ---------------------------------------------------------------------------
# 3. Loss Function (From Section 6.5.1)
# ---------------------------------------------------------------------------
class DetectionLoss(nn.Module):
    """
    Multi-component detection loss: box regression, objectness, and classification.
    """
    def __init__(self):
        super(DetectionLoss, self).__init__()
        # TODO: Define MSE for boxes, BCE for objectness/classes.
        
    def forward(self, predictions, targets):
        # Placeholder for loss calculation
        return torch.tensor(0.0, requires_grad=True)


# ---------------------------------------------------------------------------
# 4. Utility Functions
# ---------------------------------------------------------------------------
def plot_metrics(train_losses, val_losses, val_maps, output_dir="charts"):
    """Generates and saves charts for loss and mAP."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    # Loss Chart
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    # mAP Chart
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_maps, label='Validation mAP@0.5', color='green')
    plt.title('Validation Mean Average Precision (mAP@0.5)')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'map_plot.png'))
    plt.close()

def tensors_to_eval_format(pred_tensors, target_boxes, target_classes):
    """
    Converts PyTorch output tensors into your custom Prediction and BoundingBox 
    dataclasses for the eval_framework.
    """
    eval_preds, eval_gts = [], []
    # TODO: Unpack your model's tensor output into Prediction objects based on 
    # the exact tensor shapes produced by your YOLO-style head.
    return eval_preds, eval_gts


# ---------------------------------------------------------------------------
# 5. Main Training Loop
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, num_epochs=50, patience=10, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = DetectionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_losses, val_losses, val_maps = [], [], []
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, (x_motion, x_residual, targets) in enumerate(train_loader):
            x_motion, x_residual = x_motion.to(device), x_residual.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_motion, x_residual)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- VALIDATE ---
        model.eval()
        running_val_loss = 0.0
        all_eval_preds, all_eval_gts = [], []
        total_latency = 0.0
        
        with torch.no_grad():
            for x_motion, x_residual, targets in val_loader:
                x_motion, x_residual = x_motion.to(device), x_residual.to(device)
                
                # Measure latency for NFR2 benchmarking
                inf_start = time.time()
                outputs = model(x_motion, x_residual)
                inf_time = time.time() - inf_start
                total_latency += inf_time
                
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                
                # Format for evaluation
                preds, gts = tensors_to_eval_format(outputs, targets['boxes'], targets['classes'])
                all_eval_preds.extend(preds)
                all_eval_gts.extend(gts)
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate evaluation metrics
        avg_latency_per_frame = total_latency / max(len(all_eval_preds), 1)
        metrics = evaluate(all_eval_preds, all_eval_gts, latency=avg_latency_per_frame)
        val_maps.append(metrics.mAP_50)
        
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mAP@0.5: {metrics.mAP_50:.4f}")
        
        # --- EARLY STOPPING ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'best_compressed_domain_model.pth')
            print("  --> Model improved and saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
                
    print("Training complete. Generating plots...")
    plot_metrics(train_losses, val_losses, val_maps)
    
    model.load_state_dict(best_model_wts)
    return model