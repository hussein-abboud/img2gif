#!filepath: src/losses/loss_composer.py

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms.functional import normalize

try:
    from pytorch_msssim import ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


class GradientLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        target_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        return torch.mean((pred_dx - target_dx) ** 2 + (pred_dy - target_dy) ** 2)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = VGG16_Weights.DEFAULT
        vgg = vgg16(weights=weights).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = pred.shape
        pred = pred.view(B * T, C, H, W)
        target = target.view(B * T, C, H, W)
        pred = normalize(pred, self.mean, self.std)
        target = normalize(target, self.mean, self.std)
        return F.mse_loss(self.vgg(pred), self.vgg(target))


class SSIMLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not HAS_SSIM:
            raise ImportError("SSIM requires pytorch_msssim. Install it with `pip install pytorch-msssim`")
        B, T, C, H, W = pred.shape
        return 1 - ssim(pred.view(B * T, C, H, W), target.view(B * T, C, H, W), data_range=1.0, size_average=True)


class LossComposer(nn.Module):
    def __init__(self,
                 use_l1: bool = True,
                 use_grad: bool = True,
                 use_perceptual: bool = True,
                 use_ssim: bool = False,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.use_l1 = use_l1
        self.use_grad = use_grad
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim and HAS_SSIM

        self.l1 = nn.L1Loss() if use_l1 else None
        self.grad = GradientLoss() if use_grad else None
        self.perceptual = PerceptualLoss() if use_perceptual else None
        self.ssim = SSIMLoss() if self.use_ssim else None

        self.weights = weights or {
            'l1': 1.0,
            'grad': 0.2,
            'perceptual': 0.2,
            'ssim': 0.1
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.use_l1:
            loss += self.weights['l1'] * self.l1(pred, target)
        if self.use_grad:
            loss += self.weights['grad'] * self.grad(pred, target)
        if self.use_perceptual:
            loss += self.weights['perceptual'] * self.perceptual(pred, target)
        if self.use_ssim:
            loss += self.weights['ssim'] * self.ssim(pred, target)
        return loss
