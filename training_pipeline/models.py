import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell. Processes one frame, carries hidden state across the sequence."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3, padding=1
        )

    def forward(self, x, h, c):
        i, f, g, o = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Detector(nn.Module):
    """
    ConvLSTM temporal encoder + small detection head.

    Input:  (batch, seq_len, 2, H, W)  - sequence of motion vector grids
    Output: (batch, H, W, 5 + num_classes)  - per-cell bbox + objectness + class scores
    """

    def __init__(self, num_classes, hidden_channels=32):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels=2, hidden_channels=hidden_channels)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 5 + num_classes, kernel_size=1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return self.head(h).permute(0, 2, 3, 1)  # (B, H, W, 5+num_classes)
