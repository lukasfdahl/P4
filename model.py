import torch
import torch.nn as nn
from data_classes import Frame


# prepare and tokenize the data for the transformer model
class VideoBackbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, video_frames):
        pass

# the actual core of the transformer model
class TransformerCore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        pass

# the predection heads that take the transformers outputs and uses them to predict a bounding box and a class
class PredictionHeads(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.bbox_head = nn.Linear(hidden_dim, 4)
        self.class_head = nn.Linear(hidden_dim, num_classes)


    def forward(self, transformer_output):
        boxes = self.bbox_head(transformer_output)
        classes = self.class_head(transformer_output)
        return boxes, classes
    

# the full trasformer model with the backbone core and head all combined.
class ObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Initialize the 3 separate parts
        self.backbone = VideoBackbone()
        self.transformer = TransformerCore()
        self.prediction_heads = PredictionHeads(hidden_dim=256, num_classes=num_classes)

    def forward(self, video_frames):
        features = self.backbone(video_frames)
        transformer_output = self.transformer(features)
        final_boxes, final_classes = self.prediction_heads(transformer_output)

        return final_boxes, final_classes