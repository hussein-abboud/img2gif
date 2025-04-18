import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class UNetSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            DepthwiseSeparableConv(3, 16),
            DepthwiseSeparableConv(16, 16)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            DepthwiseSeparableConv(32, 32)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 64)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.conv_lstm = ConvLSTMCell(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            DepthwiseSeparableConv(64 + 64, 64),
            DepthwiseSeparableConv(64, 64)
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            DepthwiseSeparableConv(32 + 32, 32),
            DepthwiseSeparableConv(32, 32)
        )

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            DepthwiseSeparableConv(16 + 16, 16),
            DepthwiseSeparableConv(16, 16)
        )

        self.out_conv = nn.Conv2d(16, 42, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        x_enc = self.pool3(e3)  # (B, 64, 8, 8)

        # Simulate 14-time step input with same encoded feature
        h, c = torch.zeros(B, 128, x_enc.size(2), x_enc.size(3), device=x.device), \
               torch.zeros(B, 128, x_enc.size(2), x_enc.size(3), device=x.device)

        for _ in range(14):  # repeat same encoding to simulate time
            h, c = self.conv_lstm(x_enc, h, c)

        b = h  # use final hidden state as bottleneck output

        d1 = self.up1(b)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        out = self.out_conv(d3)
        out = self.activation(out)
        return out.view(-1, 14, 3, 64, 64)