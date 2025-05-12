Input (batch, 9, 360, 640)
        │
        ▼
EfficientNetV2-S Encoder
    ├─ ConvBlock(9, 64) ×2
    ├─ Fused-MBConv(64, 128) ×2
    ├─ MBConv+SE(128, 256) ×3
    └─ MBConv+SE(256, 512) ×3
        │
        ▼
Feature Map (batch, 512, H/8=45, W/8=80)
        │
        ▼
TrackNet Decoder
    ├─ Upsample ×2 → ConvBlock(512, 256)
    ├─ Upsample ×2 → ConvBlock(256, 128)
    ├─ Upsample ×2 → ConvBlock(128, 64)
    └─ ConvBlock(64, 1)  # 輸出熱力圖
        │
        ▼
Output Heatmap (batch, 1, 360, 640)

### TrackNetV2 架構說明
1. Backbone（特徵提取）
   
   - 使用 EfficientNetV2-S 作為 backbone，第一層卷積 in_channels=9（支援三幀 RGB 堆疊）。
   - 輸入 shape: (batch, 9, 360, 640)
   - 輸出 shape: 約 (batch, 256, 12, 20) （空間尺寸約為輸入的 1/32）
2. Decoder（上採樣與卷積融合）
   
   - 由三層 Upsample + Conv2d + BN + ReLU + Dropout 組成，逐步將特徵圖還原至較高解析度。
   - 通道數依序為 256 → 128 → 64 → 32 → 1
   - 最後一層用 1x1 卷積和 Sigmoid，輸出單通道 heatmap。
3. forward 流程
   
   - 輸入經 backbone 提取特徵，再經 decoder 上採樣與融合，最後輸出 heatmap。

- 功能 ：負責從輸入的多通道影像（如三幀堆疊成 9 通道）中提取高階特徵。
- 結構組成 ：
  - ConvBlock(9, 64) ×2 ：兩層標準卷積區塊，將輸入通道數從 9 提升到 64，並進行初步特徵提取。
  - Fused-MBConv(64, 128) ×2 ：融合型 MBConv 卷積模組，進一步提取並壓縮特徵，提升到 128 通道。
  - MBConv+SE(128, 256) ×3 ：帶有 Squeeze-and-Excitation 的 MBConv 模組，強化通道間的特徵關聯，提升到 256 通道。
  - MBConv+SE(256, 512) ×3 ：同上，進一步提升到 512 通道，並加強特徵表達能力。
- 輸出 ：經 backbone 處理後，輸出 shape 為 (batch, 512, H/8, W/8)，即空間尺寸經過多次下採樣。
### TrackNet Decoder（解碼器）
- 功能 ：將 backbone 輸出的高階特徵圖逐步還原回原始解析度，並生成每個像素點的預測熱力圖。
- 結構組成 ：
  - Upsample ×2 → ConvBlock(512, 256) ：上採樣兩倍，通道數降為 256，恢復部分空間解析度。
  - Upsample ×2 → ConvBlock(256, 128) ：再上採樣兩倍，通道數降為 128。
  - Upsample ×2 → ConvBlock(128, 64) ：再上採樣兩倍，通道數降為 64。
  - ConvBlock(64, 1) ：最後一層卷積，將通道數降為 1，對應單通道的熱力圖輸出。
- 輸出 ：最終輸出 shape 為 (batch, 1, 360, 640)，即與原始輸入解析度一致的熱力圖，每個像素點代表預測機率。

