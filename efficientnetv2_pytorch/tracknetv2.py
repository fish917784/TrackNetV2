import torch
import torch.nn as nn
from model import EfficientNetV2
from model import efficientnetv2_s

class TrackNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnetv2_s()  # 加上括號，實例化
        
        # Decoder: 三層上採樣+卷積，最後一層卷積到1通道
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 45x80 -> 90x160
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 90x160 -> 180x320
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 180x320 -> 360x640
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1),  # 輸出單通道
            nn.Sigmoid()
        )

    def forward(self, x):
        # 只取backbone特徵，不經過分類head
        x = self.backbone.extract_features(x)
        x = self.decoder(x)
        return x

# EfficientNetV2-S需補上extract_features方法
from model import EfficientNetV2

def extract_features(self, x):
    x = self.stem(x)
    x = self.blocks(x)
    return x

EfficientNetV2.extract_features = extract_features