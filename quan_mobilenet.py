import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from .quantization import *  # assume quan_Conv2d et quan_Linear sont définis ici

class MobileNetQuan(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetQuan, self).__init__()

        # Stem (comme dans TinyVGG style)
        self.features = nn.Sequential(
            # stem conv
            quan_Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),  # (8,32,32)
            nn.ReLU(inplace=True),

            # Block 1 : expansion 2 -> depthwise -> proj (keep style simple: use quan convs pour 1x1)
            # expansion 1x1
            quan_Conv2d(in_channels=8, out_channels=16, kernel_size=1, padding=0),  # expanded -> 16 (16,32,32)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # depthwise 3x3
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # projection 1x1
            quan_Conv2d(in_channels=16, out_channels=8, kernel_size=1, padding=0),   # back to 8 (8,32,32)
            nn.BatchNorm2d(8),

            # Block 2 (downsample)
            quan_Conv2d(in_channels=8, out_channels=16, kernel_size=1, padding=0),   # expansion -> 16 (16,32,32)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16, bias=False),  # depthwise stride2 -> (16,16,16)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            quan_Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),  # projection -> 16 (16,16,16)
            nn.BatchNorm2d(16),

            # Block 3 (no downsample)
            quan_Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=0),  # expansion -> 32 (32,16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),  # depthwise (32,16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            quan_Conv2d(in_channels=32, out_channels=24, kernel_size=1, padding=0),  # projection -> 24 (24,16,16)
            nn.BatchNorm2d(24),

            # Block 4 (downsample)
            quan_Conv2d(in_channels=24, out_channels=72, kernel_size=1, padding=0),  # expansion x3 -> 72 (72,16,16)
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, kernel_size=3, stride=2, padding=1, groups=72, bias=False),  # depthwise -> (72,8,8)
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            quan_Conv2d(in_channels=72, out_channels=32, kernel_size=1, padding=0),  # projection -> 32 (32,8,8)
            nn.BatchNorm2d(32),

            # Block 5 (no downsample)
            quan_Conv2d(in_channels=32, out_channels=96, kernel_size=1, padding=0),  # expansion x3 -> 96 (96,8,8)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=False),  # depthwise (96,8,8)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            quan_Conv2d(in_channels=96, out_channels=48, kernel_size=1, padding=0),  # projection -> 48 (48,8,8)
            nn.BatchNorm2d(48),

            # Block 6 (downsample)
            quan_Conv2d(in_channels=48, out_channels=144, kernel_size=1, padding=0),  # expansion x3 -> 144 (144,8,8)
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 144, kernel_size=3, stride=2, padding=1, groups=144, bias=False),  # depthwise -> (144,4,4)
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
            quan_Conv2d(in_channels=144, out_channels=64, kernel_size=1, padding=0),  # projection -> 64 (64,4,4)
            nn.BatchNorm2d(64),

            # À ce stade on a une sortie (64,4,4) pour entrée 32x32
        )

        # Classifier - en un bloc comme TinyVGG (flatten -> lin -> relu -> dropout -> lin)
        # 64 * 4 * 4 = 1024
        self.classifier = nn.Sequential(
            quan_Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            quan_Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten exactly comme TinyVGG
        x = self.classifier(x)
        return x

def mobilenet_quan(num_classes=10):
    """Constructs a quantized MobileNet en une seule classe, style TinyVGG."""
    model = MobileNetQuan(num_classes)
    return model