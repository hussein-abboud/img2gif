# !filepath: src/model/sequence_regressor_models.py


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

class simple_UNetSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 42, 1),  # 14 frames * 3 channels
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x.view(-1, 14, 3, 64, 64)

class ConvLSTM_UNetSequenceModel(nn.Module):
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
    
class skip_UNetSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
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

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            DepthwiseSeparableConv(128, 128)
        )

        # Decoder with skip connections
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

        # Final output layer
        self.out_conv = nn.Conv2d(16, 42, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

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
    

class ResidualBlockV2(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


class ConvLSTMCellV2(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ResConvLSTMUNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualBlockV2(64),
            ResidualBlockV2(64)
        )

        self.convlstm = ConvLSTMCellV2(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1), nn.ReLU()
        )

        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.pool2(e2)
        b = self.bottleneck(e3)

        h = torch.zeros(B, 128, b.size(2), b.size(3), device=x.device)
        c = torch.zeros(B, 128, b.size(2), b.size(3), device=x.device)

        outputs = []
        for _ in range(14):  # unroll temporal steps
            h, c = self.convlstm(b, h, c)

            d2 = self.up2(h)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))

            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))

            out = self.activation(self.out_conv(d1))
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # (B, 14, 3, 64, 64)


class AttentionBlockV3(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        def reshape(x):
            x = x.view(B, self.heads, C // self.heads, H * W)
            return x.permute(0, 1, 3, 2).reshape(B * self.heads, H * W, C // self.heads)

        q, k, v = map(reshape, (q, k, v))
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.bmm(attn, v)

        out = out.view(B, self.heads, H * W, C // self.heads)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out)


class AttentionUNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        self.attn = AttentionBlockV3(128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1), nn.ReLU()
        )

        self.out_conv = nn.Conv2d(16, 42, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                      # [B, 32, 64, 64]
        e2 = self.enc2(self.pool1(e1))         # [B, 64, 32, 32]
        e3 = self.enc3(self.pool2(e2))         # [B, 128, 16, 16]

        b = self.attn(self.pool3(e3))          # [B, 128, 8, 8]

        d1 = self.up1(b)                       # [B, 64, 16, 16]
        d1 = torch.cat([d1, e3], dim=1)        # [B, 192, 16, 16]
        d1 = self.dec1(d1)                     # [B, 64, 16, 16]

        d2 = self.up2(d1)                      # [B, 32, 32, 32]
        d2 = torch.cat([d2, e2], dim=1)        # [B, 96, 32, 32]
        d2 = self.dec2(d2)                     # [B, 32, 32, 32]

        d3 = self.up3(d2)                      # [B, 16, 64, 64]
        d3 = torch.cat([d3, e1], dim=1)        # [B, 48, 64, 64]
        d3 = self.dec3(d3)                     # [B, 16, 64, 64]

        out = self.activation(self.out_conv(d3))  # [B, 42, 64, 64]
        return out.view(-1, 14, 3, 64, 64)



class PatchEmbedV4(nn.Module):
    def __init__(self, in_channels: int = 128, patch_size: int = 8, emb_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)  # [B, emb_dim, H//patch, W//patch]
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x, (H, W)


class TransformerBlockV4(nn.Module):
    def __init__(self, emb_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(emb_dim * mlp_ratio), emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTUNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )

        self.patch_embed = PatchEmbedV4(128)
        self.transformer = nn.Sequential(
            TransformerBlockV4(),
            TransformerBlockV4()
        )
        self.unpatch = nn.ConvTranspose2d(256, 128, kernel_size=8, stride=8)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU()
        )

        self.final = nn.Conv2d(32, 42, 1)  # 14 frames × 3 channels
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)              # [B, 32, 64, 64]
        e2 = self.enc2(e1)             # [B, 64, 32, 32]
        e3 = self.enc3(e2)             # [B, 128, 16, 16]

        tokens, (h, w) = self.patch_embed(e3)
        tokens = self.transformer(tokens)
        t_out = tokens.transpose(1, 2).view(x.size(0), 256, h, w)

        b = self.unpatch(t_out)        # [B, 128, 16, 16]

        d1 = self.up1(b)               # [B, 64, 32, 32]
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)              # [B, 32, 64, 64]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.activation(self.final(d2))     # [B, 42, 64, 64]
        return out.view(-1, 14, 3, 64, 64)

