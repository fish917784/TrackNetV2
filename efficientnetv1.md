### EfficientNetV1
1.Input
  ↓
2.Stem (Conv3x3)
  ↓
3.多層 MBConv Block（每層深度、寬度、解析度根據複合縮放調整）
  ↓
4.Head (Conv1x1, Pooling, FC)
  ↓
5.Output

### EfficientNet 架構特點
1. MBConv 卷積模組
   
   - 基於 MobileNetV2 的 inverted residual block（MBConv），結合深度可分離卷積（Depthwise Separable Convolution），大幅減少參數量與計算量。
2. 複合縮放（Compound Scaling）
   
   - 傳統 CNN 通常只調整單一維度（如加深層數或加寬通道），EfficientNet 則同時調整深度、寬度與解析度，並用數學方式找到最佳平衡。
3. 多種規模（B0~B7）
   
   - EfficientNet-B0 是基礎模型，B1~B7 則是依據複合縮放策略自動擴展出來的更大模型，適用於不同運算資源需求。
4. 高效能
   
   - 在 ImageNet 等資料集上，EfficientNet 以更少的參數和 FLOPs，達到比 ResNet、VGG 等傳統架構更高的準確率。



   Input (9, 360, 640)
  ↓
EfficientNetV2-S Backbone
  ↓
(特徵圖: batch, 256, 12, 20)
  ↓
Upsample → ConvBlock(256, 256) → ConvBlock(256, 256) → ConvBlock(256, 256)
  ↓
Upsample → ConvBlock(256, 128) → ConvBlock(128, 128)
  ↓
Upsample → ConvBlock(128, 64) → ConvBlock(64, 64) → ConvBlock(64, 256)
  ↓
Reshape & (測試時) Softmax

- 輸入層

- 輸入 shape: (batch, 9, 360, 640)
- EfficientNetV2-S Backbone

- 第一層卷積 in_channels=9，out_channels=24，stride=2，空間尺寸縮小一半
- 依序經過多個 FusedMBConv/MBConv block，通道數逐步增加，空間尺寸逐步縮小
- 最後一層輸出 shape 約為 (batch, 256, 12, 20)（360/32=11.25，640/32=20）
- 上採樣與卷積融合

- 每次 Upsample（通常 scale_factor=2），接多層 ConvBlock，通道數逐步減少
- 這部分負責將特徵圖還原至較高解析度並融合細節
- 輸出層

- 最後 reshape 成 (batch, 256, H*W)
- 測試時加 softmax，得到每個像素點屬於球體的機率分布


Input (9, 360, 640)
  ↓
【EfficientNetV2-S Backbone】
  - Stem：Conv2d(9, 24, kernel=3, stride=2, padding=1) + BN + SiLU
  - Block1：FusedMBConv, repeats=2, in_c=24, out_c=24, stride=1
  - Block2：FusedMBConv, repeats=4, in_c=24, out_c=48, stride=2
  - Block3：FusedMBConv, repeats=4, in_c=48, out_c=64, stride=2
  - Block4：MBConv, repeats=6, in_c=64, out_c=128, stride=2, SE
  - Block5：MBConv, repeats=9, in_c=128, out_c=160, stride=1, SE
  - Block6：MBConv, repeats=15, in_c=160, out_c=256, stride=2, SE
  ↓
【特徵圖輸出】（通常空間尺寸縮小至約1/32，通道數256）
  ↓
Upsample (x2) → ConvBlock(256, 256) → ConvBlock(256, 256) → ConvBlock(256, 256)
  ↓
Upsample (x2) → ConvBlock(256, 128) → ConvBlock(128, 128)
  ↓
Upsample (x2) → ConvBlock(128, 64) → ConvBlock(64, 64) → ConvBlock(64, 256)
  ↓
Reshape（展平成 (batch, 256, H*W)）
  ↓
(測試時) Softmax
