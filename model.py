import torch
import torch.nn as nn
from data_classes import Frame


# tokenize the data for the transformer model (block size is the size of the blocks used for the motion vectors in the h264 video)
class H264Tokenizer(nn.Module):
    def __init__(self, mv_channels, residual_channels, hidden_dim, block_size=16):
        super().__init__()

        # Downsamples the size of the reciduals to be the size of the motion vectors (like a 256x256px video will have 256x256 reciduals but maybe only 32x32 motionvectors. this scales the reciduals down to also be 32x32)
        self.residual_downsample = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels, kernel_size=block_size, stride=block_size)

        # Used to take our motion vectors and reciduals and transfrom them into a specific number of dimensions needed by the transformer
        total_input_channels = mv_channels + residual_channels
        self.projection = nn.Conv2d(total_input_channels, hidden_dim, kernel_size=1)

    def forward(self, motion_vectors, residuals):
        #scale down the reciduals
        small_residuals = self.residual_downsample(residuals)

        #combine the motion vectors and reciduals into one tensor, by just adding the recidual channels onto the end of it.
        combined_features = torch.cat([motion_vectors, small_residuals], dim=1)

        # Project the features to have the required number of channels (256 channels)
        tokens = self.projection(combined_features)

        # Flatten the 2D grid into a 1D sequence for the Transformer
        # Transforms shape from [batch, hidden_dim, H, W] to [batch, Sequence_Length, hidden_dim]
        tokens = tokens.flatten(2).permute(0, 2, 1)
        return tokens

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
        self.tokenizer = H264Tokenizer(4, 3, 256, 16)
        self.transformer = TransformerCore()
        self.prediction_heads = PredictionHeads(hidden_dim=256, num_classes=num_classes)

    def forward(self, motion_vectors, residuals):
        features = self.tokenizer(motion_vectors, residuals)
        transformer_output = self.transformer(features)
        final_boxes, final_classes = self.prediction_heads(transformer_output)

        return final_boxes, final_classes